# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copyright 2022 XTus Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
import io

from typing import Dict, Union, Optional

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.model import modules_multimer
from alphafold.model import modules_multimer_noresample
from alphafold.model import prng
from alphafold.relax import relax
from alphafold.data import msa_pairing;
from alphafold.data import parsers;
from alphafold.data import feature_processing;
from alphafold.model import utils;
import numpy as np
import jax;
import jax.numpy as jnp;
import gzip;
import copy;

from alphafold.model import data
# Internal import (7716).

logging.set_verbosity(logging.INFO)

print(sys.argv);
sys.stdout.flush();
flags.DEFINE_list(
    'input_files', None, 'Paths to input files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')
    
flags.DEFINE_list('output_prefix', [], '<output_prefix>.(unrelaxed).pdb, <output_prefix>.metrics.pkl.gz, <output_prefix>.scores.json, <output_prefix>.timings.json will be created.'
 'The number of elements must be the same with the number of input files.'
 'The parameter file must be one.')

flags.DEFINE_string('data_dir', 'alphafold/model', 'Path to directory of supporting data.')

flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', 'jackhmmer',
                    'Path to the JackHMMER executable.')
                    
flags.DEFINE_string('hmmsearch_binary_path', 'hmmsearch',
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', 'hmmbuild',
                    'Path to the hmmbuild executable.')
                    
flags.DEFINE_string('hhblits_binary_path', 'hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', 'hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('kalign_binary_path', 'kalign',
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', 'uniref90.fasta', 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', 'mgy_clusters_2018_12.fa', 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', 'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt' , 'Path to the BFD '
                    'database for use by HHblits.')
#flags.DEFINE_string('small_bfd_database_path', 'bfd-first_non_consensus_sequences.fasta', 'Path to the small '
#                    'version of BFD used with the "reduced_dbs" preset.')

flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniclust30_database_path', 'uniclust30_2018_08', 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', 'uniprot_sprot_trembl_20220210.fasta', 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('pdb70_database_path', 'pdb70', 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', 'pdb_seqres.txt', 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', 'mmCIF', 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', 'obsolete.dat', 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
                    
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer','multimer_ic','multimer_sep','sep','monomer_ic'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 1, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                     'have been written to disk instead of running the MSA '
                     'tools. The MSA files are looked up in the output '
                     'directory, so it must stay the same between multiple '
                     'runs that are to reuse the MSAs. WARNING: This will not '
                     'check if the sequence, database or configuration have '
                     'changed.')
flags.DEFINE_boolean('run_relax', True, 'Whether to run the final relaxation '
                     'step on the predicted models. Turning relax off might '
                     'result in predictions with distracting stereochemical '
                     'violations but might help in case you are having issues '
                     'with the relaxation stage.')
flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')

flags.DEFINE_boolean('save_prevs', False, 'Save results of each recycling step.')
flags.DEFINE_boolean('gzip_features', False, 'Treat feature pickles with gzipped.')
flags.DEFINE_boolean('save_metrics',True,'Save metrics pkl file.')
flags.DEFINE_boolean('save_checkpoint',False,'Save checkpoint pkl file.')
flags.DEFINE_string('checkpoint_file',None,'Run with saved checkpoint.(You must provide correct model name manually.)')

flags.DEFINE_string('out_relax_only',None,'Only perform relax protocol and save the result as the file with specified path.');


flags.DEFINE_integer('num_recycle',3, 'Number of recycling.');
flags.DEFINE_integer('msa_crop_size',None, 'features.MSA_CROP_SIZE');
flags.DEFINE_integer('num_extra_msa',None, 'max_extra_msa or num_extra_msa in config.');


flags.DEFINE_list('model_names', None, 'Names of models to use.')
flags.DEFINE_list('model_paths', None, 'Path to the model files.')
flags.DEFINE_string('model_config_name', None, 'Name of the config to use.')
flags.DEFINE_string('custom_model_type', None, 'Name of the custom model.[template_single|template_prev_single|prev_single|template_prev_scoring]')


FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')

import gzip,re;
import pickle as pkl;

def load_pkl(filename):
  if ".gz" in filename:
    with gzip.open(filename, mode='rb') as fin:
      ret =  pkl.load(fin);
    return ret;
  else:
    with open(filename, mode='rb') as fin:
      ret =  pkl.load(fin);
    return ret;

def aatype_to_seq(aatype):
  restypes = [
  'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
  'S', 'T', 'W', 'Y', 'V', 'X', '-'
  ];
  aas = [];
  for aa in list(aatype):
    aas.append(restypes[aa]);
  return "".join(aas);
  

def create_feature_dict_single(a3m_strs):
    all_chain_features = {};
    for ii in range(len(a3m_strs)):
        in_a3m = parsers.parse_a3m(a3m_strs[ii]);
        
        input_sequence = in_a3m.sequences[0]
        input_description = in_a3m.descriptions[0]
        
        num_res = len(input_sequence)
        # Construct a default template with all zeros.
        template_features = {
            'template_aatype': np.zeros(
                (1, num_res, len(residue_constants.restypes_with_x_and_gap)),
                np.float32),
            'template_all_atom_masks': np.zeros(
                (1, num_res, residue_constants.atom_type_num), np.float32),
            'template_all_atom_positions': np.zeros(
                (1, num_res, residue_constants.atom_type_num, 3), np.float32),
            'template_domain_names': np.array([''.encode()], dtype=object),
            'template_sequence': np.array([''.encode()], dtype=object),
            'template_sum_probs': np.array([0], dtype=np.float32)
        }
        
        templates_result = templates.TemplateSearchResult(features=template_features, errors=[], warnings=[])
        
        num_res = len(re.sub(r"[\s]","",input_sequence));
        sequence_features = pipeline.make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res)
        msa_features = pipeline.make_msa_features((in_a3m,));
        
        chain_features = {**sequence_features, **msa_features, **templates_result.features};
        if len(a3m_strs) > 1:
            #msa = msa.truncate(max_seqs=self._max_uniprot_hits)
            all_seq_features = pipeline.make_msa_features([in_a3m])
            valid_feats = msa_pairing.MSA_FEATURES + (
                'msa_uniprot_accession_identifiers',
                'msa_species_identifiers',
            )
            feats = {f'{k}_all_seq': v for k, v in all_seq_features.items()
                     if k in valid_feats}
            chain_features.update(feats);
        chain_id = str(ii);
        chain_features = pipeline_multimer.convert_monomer_features(chain_features,
                         chain_id=chain_id)
        all_chain_features[chain_id] = chain_features
        
    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    np_example = feature_processing.pair_and_merge(
        all_chain_features=all_chain_features
    )

    # Pad MSA to avoid zero-sized extra_msa.
    np_example = pipeline_multimer.pad_msa(np_example, 2) #1 だとなんか不具合があると嫌なので 2

    del in_a3m;

    return np_example

def get_processed_features(feat,model_runner,random_seed=None):
  
  if random_seed is not None:
    safe_key = prng.SafeKey(jax.random.PRNGKey(random_seed+1));
  else:
    safe_key = prng.SafeKey(jax.random.PRNGKey());
    
  new_processed_feature_orig = model_runner.process_features(feat, random_seed=random_seed);
  new_processed_feature_orig['msa_profile'] = modules_multimer.make_msa_profile(new_processed_feature_orig);
  
  if 'prev_pos' in feat:
    new_processed_feature_orig['prev_pos'] = feat['prev_pos'];
    
  if 'orig_atom_positions' in feat:
    new_processed_feature_orig['orig_atom_positions'] = feat['orig_atom_positions'];
    
  msa_feat = [];
  msa_mask = [];
  extra_msa_feat = [];
  extra_msa_mask = [];
  bert_mask = [];
  true_msa = [];
  current_keys = list(new_processed_feature_orig.keys());
  for z in range(model_runner.config.model.num_recycle+1):
    safe_key, sample_key, mask_key = safe_key.split(3);
    new_processed_feature = copy.deepcopy(new_processed_feature_orig);
    #ここ modules_multimer_scoring を使わないで良いか？
    new_processed_feature = modules_multimer_noresample.sample_msa(sample_key, new_processed_feature, model_runner.config.model.embeddings_and_evoformer.num_msa)
    for k in ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']:
      if k in new_processed_feature:
        new_processed_feature['extra_' + k] = new_processed_feature[k]


    new_processed_feature = modules_multimer.make_masked_msa(new_processed_feature, mask_key, model_runner.config.model.embeddings_and_evoformer.masked_msa)
    
    (new_processed_feature['cluster_profile'],
     new_processed_feature['cluster_deletion_mean']) = modules_multimer.nearest_neighbor_clusters(new_processed_feature)
    
    msa_feat.append(modules_multimer.create_msa_feat(new_processed_feature));
    msa_mask.append(new_processed_feature["msa_mask"]);
     
    (extra_msa_feat_,
    extra_msa_mask_) = modules_multimer.create_extra_msa_feature(new_processed_feature
    , model_runner.config.model.embeddings_and_evoformer.num_extra_msa)
    extra_msa_feat.append(extra_msa_feat_);
    extra_msa_mask.append(extra_msa_mask_);
    bert_mask.append(new_processed_feature["bert_mask"]);
    true_msa.append(new_processed_feature["true_msa"]);
    del extra_msa_feat_;
    del extra_msa_mask_;
    
  del new_processed_feature_orig;
  
  non_ensembled_features = {};
  ensembled_features = {};
  for cc in current_keys:
    if cc in ['msa','deletion_matrix','bert_mask','msa_mask','msa_profile','cluster_bias_mask'] :
      continue;
    non_ensembled_features[cc] = new_processed_feature[cc];
  
  del new_processed_feature;
  
  ensembled_features["msa_mask"] = np.array(msa_mask);
  ensembled_features["msa_feat"] = np.array(msa_feat);
  ensembled_features["bert_mask"] = np.array(bert_mask);
  ensembled_features["true_msa"] = np.array(true_msa);
  ensembled_features["extra_msa_feat"] = np.array(extra_msa_feat);
  ensembled_features["extra_msa_mask"] = np.array(extra_msa_mask);
  
  
  """
  for kk in ensembled_features.keys():
    if "shape" in dir(ensembled_features[kk]):
      print("ensembled_features",kk,ensembled_features[kk].shape);
  for kk in non_ensembled_features.keys():
    if "shape" in dir(non_ensembled_features[kk]):
      print("non_ensembled_features",kk,non_ensembled_features[kk].shape);
  """
  return {"ens":ensembled_features,"nonens":non_ensembled_features};


def create_features_from_pdb(pdb_file,insert_prev,dup_prev,scoring):
  if dup_prev:
    assert not insert_prev;
    
  lines_ = [];
  if re.search("\.gz",pdb_file):
    with gzip.open(pdb_file,"rt") as fin:
      lines_ = fin.readlines();
  else:
    with open(pdb_file,"rt") as fin:
      lines_ = fin.readlines();
  
  used_chain_ids_all = [];
  ret_prop_all = [];
  chain_ter = {};
  lines_all = [];
  lines = {};
  mseflag = False;
  has_a = False;
  for ii in range(len(lines_)):
    if re.search("^TER",lines_[ii]):
      if len(lines_[ii]) > 21:
        chain_ter[lines_[ii][21]] = 100;
    if re.search("^ENDMDL",lines_[ii]):
      chain_ter = {};
      if len(lines) > 0:
        lines_all.append(lines);
        lines = {};
    if re.search("^HETATM",lines_[ii]) or re.search("^ATOM",lines_[ii]):
      chainid = lines_[ii][21];
      if chainid == "A":
          has_a = True;
      if not chainid in lines:
        lines[chainid] = [];
        
      if lines_[ii][21] in chain_ter:
        continue;
      if lines_[ii][17:20] == "MSE":
        llis = list(lines_[ii]);
        llis[17:20] = "MET";
        if lines_[ii][12:16] == "SE  ":
          llis[12:16] = " SD ";
        lines_[ii] = "".join(llis);
        mseflag = True;
      lines[chainid].append(lines_[ii]);
  if mseflag:
    sys.stderr.write("MSE was changed to MET.\n");
  if len(lines) > 0:
    lines_all.append(lines);
    
  allchains = [];
  entities = {};
  asymcount = 0;
  
  chain_check = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  
  for lines in list(lines_all):
    for cc in list(sorted(lines.keys())):
      prot_pdb = protein.from_pdb_string("".join(lines[cc]));
      sseq = aatype_to_seq(prot_pdb.aatype);
      if not sseq in entities:
        entities[sseq] = [0,len(entities)+1];
      
      entities[sseq][0] += 1;
      asymcount+=1;
      
      entityid = entities[sseq][1];
      symid = entities[sseq][0];
      asymid = asymcount;
      
      seqlen = len(sseq);
      
      allchains.append(
      {
      "aatype": prot_pdb.aatype,
      "atom_positions": prot_pdb.atom_positions,
      "atom_mask": prot_pdb.atom_mask,
      "residue_index": prot_pdb.residue_index,# この辺一応作ってはいるが使っていない
      "entity_id": [entityid]*seqlen,
      "sym_id": [symid]*seqlen,
      "asym_id": [asymid]*seqlen
      });
      
  
  all_chain_features = {};
  t_aatypes = [];
  t_masks = [];
  t_pos = [];
  t_index = [];
  a3ms = [];
  for ii in range(len(allchains)):
    chain = allchains[ii];
    
    chain_id = str(chain["asym_id"][0]);
    a3ms.append(">seq"+str(ii)+"\n"+aatype_to_seq(chain["aatype"])+"\n");
    t_aatypes.append(chain["aatype"]);
    t_masks.append(chain["atom_mask"]);
    t_pos.append(chain["atom_positions"]);
    t_index.append(chain["residue_index"]);

  np_example = create_feature_dict_single(a3ms);
  
  chk = np.concatenate(t_aatypes,axis=0);
  rindexx = np.concatenate(t_index,axis=0);
  
  if (np.concatenate(t_aatypes,axis=0) == np_example["aatype"]).all():
      #同一配列の場合組み換えが起こっている可能性がある
      assert rindexx.shape == np_example["residue_index"].shape;
      rindexx = rindexx-1;
      np_example["residue_index"] = rindexx;
  else:
      sys.stderr.write("Can not re-assign residue index because there are duplicate sequences.\n Please re-assign it with other scripts if they are truncated.\n");
      sys.stderr.flush();
      
      
  if not insert_prev:
    np_example['num_templates'] = np.int32(1);
    ############### ここ間違い！！！！！！！！！！！！！！！！！！！！！！！！ template_aatype は HHBLITS_AA_TO_ID で別のマッピング
    np_example['template_aatype'] = np.concatenate(t_aatypes,axis=0)[None,:];
    np_example['template_all_atom_mask'] = np.concatenate(t_masks,axis=0)[None,:];
    np_example['template_all_atom_positions'] = np.concatenate(t_pos,axis=0)[None,:];
    assert (np_example['template_aatype'][0] == np_example['aatype']).all();
    if dup_prev:
      np_example['prev_pos'] = np.concatenate(t_pos,axis=0);
  else:
    np_example['num_templates'] = np.int32(0);
    
    del np_example['template_aatype'];
    del np_example['template_all_atom_mask'];
    del np_example['template_all_atom_positions'];
    
    np_example['prev_pos'] = np.concatenate(t_pos,axis=0);
    
  if scoring:
    np_example['orig_atom_positions'] = np.concatenate(t_pos,axis=0);
    
  return np_example;

def relax_only(amber_relaxer,pdb_file,output_path):
  with open(pdb_file) as fin:
    lines = fin.readlines();
  prot_pdb = protein.from_pdb_string("".join(lines));
  relaxed_pdb_str, _, _ = amber_relaxer.process(prot=prot_pdb)
  
  with open(output_path, 'w') as f:
    f.write(relaxed_pdb_str)



def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    gzip_features:bool=False,
    no_resample=False,
    insert_prev=False,
    dup_prev=False,
    scoring=False,
    output_prefix:str=None,
    save_metrics=True,
    save_checkpoint=False,
    checkpoint_file=None,
    ):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  
  if output_prefix is not None:
    assert len(model_runners) == 1, "output_prefix can predict with one model.";
    
  if checkpoint_file is not None:
    assert len(model_runners) == 1, "checkpoint_file can predict with one model.";
    
  timings = {}
  
  #output_prefix がある場合必要ないが、何か不具合があるといけないので一応与えて欲しい
  # Todo if で場合分けする
  
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Write out features as a pickled dictionary.
  
  if output_prefix is not None:
    features_output_path = output_prefix+ '.features.pkl';
  else:
    features_output_path = os.path.join(output_dir, 'features.pkl');
    
  if(os.path.exists(features_output_path)):
    sys.stderr.write("!!!!!!!!!!!!Precomputed features will be used "+features_output_path);
    sys.stderr.write("\n");
    t_0 = time.time()
    feature_dict = load_pkl(features_output_path);
    timings['features'] = time.time() - t_0
  else:
    if re.search(r"\.pkl(\.gz)?$",fasta_path):
      t_0 = time.time()
      feature_dict = load_pkl(fasta_path);
      timings['features'] = time.time() - t_0
    elif re.search(r"\.pdb$",fasta_path):
      t_0 = time.time()
      feature_dict = create_features_from_pdb(fasta_path,insert_prev=insert_prev,dup_prev=dup_prev,scoring=scoring);
      timings['features'] = time.time() - t_0
    else:
      # Get features.
      t_0 = time.time()
      if output_prefix is not None:
        is_a3m = re.search("\.a3m(\.gz)?",fasta_path);
        if not is_a3m:
          raise Exception("This mode does not support msa construction.");
          
      feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)

      timings['features'] = time.time() - t_0

      # Write out features as a pickled dictionary.
      if gzip_features:
        with gzip.open(features_output_path+'.gz', 'wb') as f:
          pickle.dump(feature_dict, f, protocol=4)
      else:
        with open(features_output_path, 'wb') as f:
          pickle.dump(feature_dict, f, protocol=4)
          
  unrelaxed_pdbs = {}
  relaxed_pdbs = {}
  ranking_confidences = {}
  checkpoint_features = None;
  if checkpoint_file is not None:
    checkpoint_features = load_pkl(checkpoint_file);
    
  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    if no_resample:
      processed_feature_dict = get_processed_features(feature_dict,model_runner,model_random_seed);
      processed_feature_dict["nonens"]['ptm_bin_centers'] = confidence._calculate_bin_centers(jnp.linspace(0., model_runner.config.model.heads.predicted_aligned_error.max_error_bin, model_runner.config.model.heads.predicted_aligned_error.num_bins - 1))
    else:
      processed_feature_dict = model_runner.process_features(
          feature_dict, random_seed=model_random_seed)
      processed_feature_dict['ptm_bin_centers'] = confidence._calculate_bin_centers(jnp.linspace(0., model_runner.config.model.heads.predicted_aligned_error.max_error_bin, model_runner.config.model.heads.predicted_aligned_error.num_bins - 1))
    timings[f'process_features_{model_name}'] = time.time() - t_0
    
    if checkpoint_features is not None:
      if no_resample:
        processed_feature_dict["nonens"]["prev_pos"] = checkpoint_features["pos"];
        processed_feature_dict["nonens"]["prev_msa_first_row"] = checkpoint_features["msa_first_row"];
        processed_feature_dict["nonens"]["prev_pair"] = checkpoint_features["pair"];
      else:
        processed_feature_dict["prev_pos"] = checkpoint_features["pos"];
        processed_feature_dict["prev_msa_first_row"] = checkpoint_features["msa_first_row"];
        processed_feature_dict["prev_pair"] = checkpoint_features["pair"];
        
    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict,
                                             random_seed=model_random_seed)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, t_diff)

    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict,
                           random_seed=model_random_seed)
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
          model_name, fasta_name, t_diff)
    if no_resample:
      processed_feature_dict = {**processed_feature_dict["ens"],**processed_feature_dict["nonens"]};
      
    plddt = prediction_result['plddt']
    if len(prediction_result['ranking_confidence'].shape) == 0:
        ranking_confidences[model_name] = float(prediction_result['ranking_confidence'])
    else:
        if len(prediction_result['ranking_confidence'].shape) > 1 or prediction_result['ranking_confidence'].shape[0] > 1:
            sys.stderr.write("The shapes of ranking_confidence is larger than 1. Only the first element is used. "+str(prediction_result['ranking_confidence']));
        ranking_confidences[model_name] = float(prediction_result['ranking_confidence'][0])
    print(prediction_result['ranking_confidence']);
    
    if model_runner.save_prevs:
      prevs = prediction_result['prevs'];
      pnum = prevs['pos'].shape[0];
      dummybuff = copy.deepcopy(prediction_result);
      
      for pp in range(pnum):
        if output_prefix is not None:
          out_pdb_path = output_prefix+(f'.recycling.{pp}.pdb');
          out_pkl_path = output_prefix+(f'.recycling.{pp}.metrics.pkl');
        else:
          out_pdb_path = os.path.join(output_dir, f'recycling_{model_name}.{pp}.pdb');
          out_pkl_path = os.path.join(output_dir, f'recycling_{model_name}.{pp}.metrics.pkl');
        
        dummybuff['structure_module']['final_atom_positions'] = prevs['pos'][pp];
      
      for pp in range(pnum):
        if output_prefix is not None:
          out_pdb_path = output_prefix+(f'.recycling.{pp}.pdb');
        else:
          out_pdb_path = os.path.join(output_dir, f'recycling_{model_name}.{pp}.pdb');
        dummybuff['structure_module']['final_atom_positions'] = prevs['pos'][pp];
        sys.stderr.flush();
        print("recycle:",pp,"file:",out_pdb_path,"metrics(plddt):",np.mean(prevs['prevs_plddt'][pp]));
        if 'prevs_iptm' in prevs:
          print("recycle:",pp,"file:",out_pdb_path,"metrics(iptm):",prevs['prevs_iptm'][pp]);
        if 'prevs_ptm' in prevs:
          print("recycle:",pp,"file:",out_pdb_path,"metrics(ptm):",prevs['prevs_ptm'][pp]);
        sys.stdout.flush();
        
        out_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=dummybuff,
        b_factors= np.repeat(prevs['prevs_plddt'][pp,:, None], residue_constants.atom_type_num, axis=-1),
        remove_leading_feature_dimension=not model_runner.multimer_mode);
        with open(out_pdb_path, 'w') as f:
          f.write(protein.to_pdb(out_protein));
        
        del out_protein;
      if 'predicted_aligned_error_breaks' in prediction_result:
        del prediction_result['predicted_aligned_error_breaks']
      del prevs;
      del prediction_result['prevs'];
      del dummybuff;
    
    elif False:
      # RECYCLE が大きいと OOM になるので一旦保留
      # Save results of each recycling step.
      prevs = prediction_result['prevs'];
      pnum = prevs['pos'].shape[0];

      dummybuff = copy.deepcopy(prediction_result);
      
      for pp in range(pnum):
        if output_prefix is not None:
          out_pdb_path = output_prefix+(f'.recycling.{pp}.pdb');
          out_pkl_path = output_prefix+(f'.recycling.{pp}.metrics.pkl');
        else:
          out_pdb_path = os.path.join(output_dir, f'recycling_{model_name}.{pp}.pdb');
          out_pkl_path = os.path.join(output_dir, f'recycling_{model_name}.{pp}.metrics.pkl');
        
        dummybuff['structure_module']['final_atom_positions'] = prevs['pos'][pp];
        if 'predicted_aligned_error' in dummybuff:
          dummybuff['predicted_aligned_error'] = {};
          dummybuff['predicted_aligned_error']['logits'] = prevs['predicted_aligned_error_logits'][pp];
          dummybuff['predicted_aligned_error']['breaks'] = dummybuff['predicted_aligned_error_breaks'];
          dummybuff['predicted_aligned_error']['asym_id'] = processed_feature_dict['asym_id'];
        dummybuff['predicted_lddt']['logits'] = prevs['predicted_lddt_logits'][pp];

        cres = model.get_confidence_metrics(dummybuff, multimer_mode=model_runner.multimer_mode);
        
        
        sys.stderr.flush();
        print("recycle:",pp,"file:",out_pdb_path,"metrics(plddt):",np.mean(cres['plddt']));
        if 'iptm' in cres:
          print("recycle:",pp,"file:",out_pdb_path,"metrics(iptm):",cres['iptm']);
        if 'ptm' in cres:
          print("recycle:",pp,"file:",out_pdb_path,"metrics(ptm):",cres['ptm']);
        sys.stdout.flush();
        
        out_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=dummybuff,
        b_factors= np.repeat(cres['plddt'][:, None], residue_constants.atom_type_num, axis=-1),
        remove_leading_feature_dimension=not model_runner.multimer_mode);
        if save_metrics:
          if gzip_features:
            with gzip.open(out_pkl_path+'.gz','wb') as f:
              pickle.dump(cres,f,protocol=4);
          else:
            with open(out_pkl_path,'wb') as f:
              pickle.dump(cres,f,protocol=4);
          
        with open(out_pdb_path, 'w') as f:
          f.write(protein.to_pdb(out_protein));
        
        del cres;
        del out_protein;
      if 'predicted_aligned_error_breaks' in prediction_result:
        del prediction_result['predicted_aligned_error_breaks']
      del prevs;
      del prediction_result['prevs'];
      del dummybuff;

    # Save the model outputs.
    if output_prefix is not None:
      result_output_path = output_prefix+".metrics.pkl";
      result_checkpoint_path = output_prefix+".checkpoint.pkl";
    else:
      result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl');
      result_checkpoint_path = os.path.join(output_dir, f'result_{model_name}.checkpoint.pkl');
      
    if save_metrics:
      if gzip_features:
        with gzip.open(result_output_path+'.gz', 'wb') as f:
          pickle.dump(prediction_result, f, protocol=4)
      else:
        with open(result_output_path, 'wb') as f:
          pickle.dump(prediction_result, f, protocol=4)
    
    if save_checkpoint:
      checkpoint = {
      'msa_first_row':prediction_result['representations']['msa_first_row'],
      'pair':prediction_result['representations']['pair'],
      'pos':prediction_result['structure_module']['final_atom_positions']
      };
      
      if gzip_features:
        with gzip.open(result_checkpoint_path+'.gz', 'wb') as f:
          pickle.dump(checkpoint, f, protocol=4)
      else:
        with open(result_checkpoint_path, 'wb') as f:
          pickle.dump(checkpoint, f, protocol=4)

    
    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    if output_prefix is not None:
      unrelaxed_pdb_path = output_prefix+'.unrelaxed.pdb'
    else:
      unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    sys.stderr.flush();
    print("final: final","file:",unrelaxed_pdb_path, "metrics(plddt):",np.mean(prediction_result['plddt']));
    if 'iptm' in prediction_result:
      print("final: final","file:",unrelaxed_pdb_path, "metrics(iptm):",prediction_result['iptm']);
    if 'ptm' in prediction_result:
      print("final: final","file:",unrelaxed_pdb_path, "metrics(ptm):",prediction_result['ptm']);
    sys.stdout.flush();
    
    
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

    if amber_relaxer:
      # Relax the prediction.
      t_0 = time.time()
      relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
      timings[f'relax_{model_name}'] = time.time() - t_0

      relaxed_pdbs[model_name] = relaxed_pdb_str

      # Save the relaxed PDB.
      if output_prefix is not None:
        unrelaxed_pdb_path = output_prefix+'.relaxed.pdb'
      else:
        relaxed_output_path = os.path.join(
          output_dir, f'relaxed_{model_name}.pdb')
          
      with open(relaxed_output_path, 'w') as f:
        f.write(relaxed_pdb_str)

  # Rank by model confidence and write out relaxed PDBs in rank order.
  ranked_order = []
  for idx, (model_name, _) in enumerate(
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
    if output_prefix is not None:
      ranked_output_path = output_prefix+'.pdb';
    else:
      ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
      
    with open(ranked_output_path, 'w') as f:
      if amber_relaxer:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])
        
  if output_prefix is not None:
    ranking_output_path = output_prefix+'.scores.json'
  else:
    ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))

  logging.info('Final timings for %s: %s', fasta_name, timings)

  if output_prefix is not None:
    timings_output_path = output_prefix+'.timings.json'
  else:
    timings_output_path = os.path.join(output_dir, 'timings.json')
    
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


  if FLAGS.out_relax_only is not None:
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=FLAGS.use_gpu_relax)
    assert len(FLAGS.input_files) == 1, "input_files must be one."
    assert re.search("pdb$",FLAGS.input_files[0]) is not None, "input_files must be pdb."
    relax_only(amber_relaxer,FLAGS.input_files[0],FLAGS.out_relax_only);
    sys.exit(0);

  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', 'db_preset',
              should_be_set=use_small_bfd)
  _check_flag('bfd_database_path', 'db_preset',
              should_be_set=not use_small_bfd)
  _check_flag('uniclust30_database_path', 'db_preset',
              should_be_set=not use_small_bfd)

  run_multimer_system = 'multimer' in FLAGS.model_preset
  
  #余分があっても気にしない
  #_check_flag('pdb70_database_path', 'model_preset',
  #            should_be_set=not run_multimer_system)
  #_check_flag('pdb_seqres_database_path', 'model_preset',
  #            should_be_set=run_multimer_system)
  #_check_flag('uniprot_database_path', 'model_preset',
  #            should_be_set=run_multimer_system)

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1
    
  is_a3m = False;
  if re.search(r"\.a3m(\.gz)?$",FLAGS.input_files[0]):
    is_a3m = True;
    
  if is_a3m:
    fasta_names = [pathlib.Path(FLAGS.input_files[0]).stem];
  else:
    fasta_names = [pathlib.Path(p).stem for p in FLAGS.input_files]
  
  if len(FLAGS.output_prefix) == 0:
    # Check for duplicate FASTA file names.
    if len(fasta_names) != len(set(fasta_names)):
      raise ValueError('All FASTA paths must have a unique basename.')

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=FLAGS.pdb_seqres_database_path)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[FLAGS.pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas)

  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas)
  else:
    num_predictions_per_model = 1
    data_pipeline = monomer_data_pipeline

  model_runners = {};
  model_name_map = None;
  if FLAGS.model_preset == "sep" or FLAGS.model_preset == "multimer_sep":
    if FLAGS.model_names is not None:
      model_names = FLAGS.model_names;
    elif  FLAGS.model_paths is not None:
      model_name_map = {};
      model_names = [];
      for pp in list(FLAGS.model_paths):
        nname = re.sub(r"\.npz","",re.sub(r".*[\\/]","",pp));
        if nname in model_name_map:
          raise Exception("Name of the files should be unique.\n"+str(FLAGS.model_paths));
        model_name_map[nname] = pp;
        model_names.append(nname);
    else:
      raise Exception("--model_names or --model_paths are required.");
  else:
    model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  
  no_resample = False;
  insert_prev=False;# PDB をインプットにするモード用
  dup_prev=False;# PDB をインプットにするモード用
  for_scoring=False;
  
  if FLAGS.custom_model_type is not None:
    assert FLAGS.model_config_name is None,"model_config_name can not use with custom_model_type";
    if FLAGS.custom_model_type == "template_single":
      model_config = config.model_config("model_1_multimer_v2")
      model_config.model.embeddings_and_evoformer.template.enabled = True;
      model_config.model.embeddings_and_evoformer.num_extra_msa = 2;
      model_config.model.embeddings_and_evoformer.num_msa = 1;
      model_config.model.resample_msa_in_recycling = False;
      no_resample=True;
    elif FLAGS.custom_model_type == "template_prev_single":
      model_config = config.model_config("model_1_multimer_v2")
      model_config.model.embeddings_and_evoformer.template.enabled = True;
      model_config.model.embeddings_and_evoformer.num_extra_msa = 2;
      model_config.model.embeddings_and_evoformer.num_msa = 1;
      model_config.model.resample_msa_in_recycling = False;
      dup_prev=True;
      no_resample=True;
    elif FLAGS.custom_model_type == "prev_single":
      model_config = config.model_config("model_1_multimer_v2")
      model_config.model.embeddings_and_evoformer.template.enabled = False;
      model_config.model.embeddings_and_evoformer.num_extra_msa = 2;
      model_config.model.embeddings_and_evoformer.num_msa = 1;
      model_config.model.resample_msa_in_recycling = False;
      insert_prev=True;
      no_resample=True;
    elif FLAGS.custom_model_type == "template_prev_scoring":
      model_config = config.model_config("model_1_multimer_v2")
      model_config.model.embeddings_and_evoformer.template.enabled = True;
      model_config.model.embeddings_and_evoformer.num_extra_msa = 2;
      model_config.model.embeddings_and_evoformer.num_msa = 1;
      model_config.model.resample_msa_in_recycling = False;
      dup_prev=True;
      no_resample=True;
      for_scoring=True;
    else:
      raise Exception("custom_model_type must be template_single|template_prev_single|prev_single|template_prev_scoring???");
      
  for model_name in model_names:
    if FLAGS.custom_model_type is None:
      if FLAGS.model_config_name is not None:
        model_config = config.model_config(FLAGS.model_config_name)
      else:
        model_config = config.model_config(model_name);
        
    if run_multimer_system:
      model_config.model.num_recycle = FLAGS.num_recycle;
      if FLAGS.num_extra_msa is not None:
          model_config.model.embeddings_and_evoformer.num_extra_msa = FLAGS.num_extra_msa;
      if FLAGS.msa_crop_size is not None:
          alphafold.data.feature_processing.MSA_CROP_SIZE = FLAGS.msa_crop_size;
    else:
      model_config.data.common.num_recycle = FLAGS.num_recycle;
      model_config.model.num_recycle = FLAGS.num_recycle;
      if FLAGS.num_extra_msa is not None:
          data.common.max_extra_msa = FLAGS.num_extra_msa;

    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    if model_name_map is not None:
      with open(model_name_map[model_name], 'rb') as f:
        model_params = utils.flat_params_to_haiku(np.load(io.BytesIO(f.read()), allow_pickle=False))
    else:
      model_params = data.get_model_haiku_params(
          model_name=model_name, data_dir=FLAGS.data_dir)
          
    model_runner = model.RunModel(model_config, model_params
      ,save_prevs=FLAGS.save_prevs
      ,no_resample=no_resample
      ,scoring=for_scoring
      ,return_representations=FLAGS.save_checkpoint)
      
    for i in range(num_predictions_per_model):
      model_runners[f'{model_name}_pred_{i}'] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  if FLAGS.run_relax:
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=FLAGS.use_gpu_relax)
  else:
    amber_relaxer = None

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  if len(FLAGS.output_prefix) > 0:
    assert len(FLAGS.output_prefix) == len(FLAGS.input_files), 'The number of output_prefix and input_files must be the same.';
    assert len(model_runners) == 1, 'output_prefix can not be used with multiple models or num_multimer_predictions_per_model.';
  
  file_paths = FLAGS.input_files;
  if is_a3m:
    file_paths = [",".join(file_paths)];
  # Predict structure for each of the sequences.
  for i, fasta_path in enumerate(file_paths):
    fasta_name = fasta_names[i]
    preff = None;
    if len(FLAGS.output_prefix) > 0:
      preff = FLAGS.output_prefix[i];
      
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        gzip_features=FLAGS.gzip_features,
        no_resample=no_resample,
        insert_prev=insert_prev,
        dup_prev=dup_prev,
        scoring=for_scoring,
        output_prefix=preff,
        save_metrics=FLAGS.save_metrics,
        save_checkpoint=FLAGS.save_checkpoint,
        checkpoint_file=FLAGS.checkpoint_file
        )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'input_files',
      #'output_dir',
      #'data_dir',
      #'uniref90_database_path',
      #'mgnify_database_path',
      #'template_mmcif_dir',
      #'max_template_date',
      #'obsolete_pdbs_path',
      #'use_gpu_relax',
  ])

  app.run(main)
