"""
Self-contained launcher for all model-specific CoFi workflows.

Examples:
  python cofi.py --model bertbase --block 2 --task sst2
  python cofi.py --block 2 --task sst2
  python cofi.py --model tinybert --status
  python cofi.py --model mobilebert --block 4

The model-specific scripts are embedded below and executed in isolated
namespaces. Their block0/block1/etc. implementations are intentionally kept
separate, because each model patches its own incompatible CoFiPruning clone.
"""

import argparse
import subprocess
import sys


MODEL_SPECS = {
    'bertbase': {"label": 'BERT-base', "aliases": ('bert', 'base', 'bert-base', 'bert_base')},
    'tinybert': {"label": 'TinyBERT', "aliases": ('tiny', 'tiny-bert', 'tiny_bert')},
    'distilbert': {"label": 'DistilBERT', "aliases": ('distil', 'distil-bert', 'distil_bert')},
    'mobilebert': {"label": 'MobileBERT', "aliases": ('mobile', 'mobile-bert', 'mobile_bert')},
    'albert': {"label": 'ALBERT', "aliases": ('albertbase', 'albert-base', 'albert_base')},
}

MODEL_SOURCES = {'albert': '"""\n'
           'CoFiPruning — ALBERT-base-v2 on 5 GLUE tasks.\n'
           '\n'
           'Usage:\n'
           '  python cofi_albert.py --status\n'
           '  python cofi_albert.py --block 0           # fresh clone + all patches\n'
           '  python cofi_albert.py --block 1           # download all 5 fine-tuned models\n'
           '  python cofi_albert.py --block 1 --task rte\n'
           '  python cofi_albert.py --block 2           # prune all 5\n'
           '  python cofi_albert.py --block 2 --task rte\n'
           '  python cofi_albert.py --block 3           # evaluate all\n'
           '  python cofi_albert.py --block 3 --task rte\n'
           '  python cofi_albert.py --block 4           # results table\n'
           '\n'
           'Notes:\n'
           '  - BASE_DIR is the directory containing this script (relative paths throughout).\n'
           '  - Block 0 does a FRESH clone every time (deletes CoFiPruning/ if it exists).\n'
           '    This makes block 0 idempotent and avoids cross-model patch contamination.\n'
           '  - Run block 0 immediately before block 2 for a clean state.\n'
           '  - modeling_albert.py must be in the same directory as this script.\n'
           '"""\n'
           '\n'
           'import argparse\n'
           'import json\n'
           'import os\n'
           'import re\n'
           'import subprocess\n'
           'import sys\n'
           'import time\n'
           '\n'
           '# ── Paths (relative — works on any server) ────────────────────────────────────\n'
           'BASE_DIR = os.path.dirname(os.path.abspath(__file__))\n'
           "AL_REPO_DIR = os.path.join(BASE_DIR, 'CoFiPruning_Albert')\n"
           '\n'
           '# ── Constants ──────────────────────────────────────────────────────────────────\n'
           'SPARSITY    = 0.6\n'
           'SEED        = 57\n'
           "ALL_TASKS   = ['sst2', 'qnli', 'mnli', 'qqp', 'rte']\n"
           "SMALL_TASKS = {'rte', 'mrpc', 'cola', 'stsb'}\n"
           '\n'
           '# Pre-finetuned ALBERT-base-v2 checkpoints.\n'
           '# textattack/ models are popular and well-tested. Swap if you find better ones.\n'
           '# Always run --block 3 (eval unpruned) to verify quality before pruning.\n'
           'AL_PRETRAINED_FT = {\n'
           "    'sst2': 'textattack/albert-base-v2-SST-2',  #checked\n"
           "    'qnli': 'Alireza1044/albert-base-v2-qnli', #checked\n"
           "    'mnli': 'Alireza1044/albert-base-v2-mnli', #checked\n"
           "    'qqp':  'textattack/albert-base-v2-QQP', #checked\n"
           "    'rte':  'textattack/albert-base-v2-RTE', #checked\n"
           '}\n'
           '\n'
           '# Non-embedding param count for sparsity %.\n'
           '# Computed via calculate_parameters() from utils/utils.py on a fresh\n'
           '# AlbertForSequenceClassification(AlbertConfig(num_labels=2)).\n'
           '# DO NOT hand-calculate — see context doc section 9.\n'
           '# Set to None to auto-compute on first block 3 run.\n'
           'ALBERT_BASE_PARAMS = None  # will be computed if None\n'
           '\n'
           '\n'
           '# ── Path helpers ───────────────────────────────────────────────────────────────\n'
           'def ft_dir(task):\n'
           "    return os.path.join(BASE_DIR, f'ft_albert_{task}')\n"
           '\n'
           'def pr_dir(task):\n'
           "    return os.path.join(BASE_DIR, f'pr_albert_{task}_s{int(SPARSITY * 100)}')\n"
           '\n'
           'def eval_path(d):\n'
           "    return os.path.join(d, 'eval_results.json')\n"
           '\n'
           'def model_saved(path):\n'
           "    return (os.path.exists(os.path.join(path, 'pytorch_model.bin')) or\n"
           "            os.path.exists(os.path.join(path, 'model.safetensors')))\n"
           '\n'
           'def task_cfg(task):\n'
           '    if task in SMALL_TASKS:\n'
           '        return dict(prune_epochs=100, eval_steps=50, save_steps=50,\n'
           '                    prepruning=4, lag_warmup=20, layer_distill_v=4)\n'
           '    return dict(prune_epochs=20, eval_steps=500, save_steps=500,\n'
           '                prepruning=1, lag_warmup=2, layer_distill_v=3)\n'
           '\n'
           'def header(msg):\n'
           '    W = 60\n'
           "    print('\\n' + '=' * W)\n"
           "    print(f'  {msg}')\n"
           "    print('=' * W)\n"
           '\n'
           '\n'
           '# ── Patch helper ───────────────────────────────────────────────────────────────\n'
           'def patch_file(fpath, old, new, description):\n'
           '    """Idempotent string-replace patch with clear status output."""\n'
           '    if not os.path.exists(fpath):\n'
           "        print(f'  [missing file]    {description} — {fpath}')\n"
           '        return\n'
           '    txt = open(fpath).read()\n'
           '    if new and new in txt:\n'
           "        print(f'  [already patched] {description}')\n"
           '        return\n'
           '    if old not in txt:\n'
           "        print(f'  [not found]       {description}')\n"
           '        # Print surrounding context to help debug whitespace mismatches\n'
           '        for i, line in enumerate(txt.splitlines(), 1):\n'
           '            if any(frag in line for frag in old.splitlines()[:1]):\n'
           "                print(f'    hint line {i}: {repr(line)}')\n"
           '        return\n'
           "    open(fpath, 'w').write(txt.replace(old, new))\n"
           "    print(f'  [patched]         {description}')\n"
           '\n'
           '\n'
           '# ── Block 0: Fresh clone + all patches ────────────────────────────────────────\n'
           'def block0():\n'
           "    header('BLOCK 0 — Fresh clone + patch CoFiPruning for ALBERT')\n"
           '\n'
           '    # ── Fresh clone ────────────────────────────────────────────────────────────\n'
           '    if os.path.exists(AL_REPO_DIR):\n'
           "        print(f'Removing existing {AL_REPO_DIR} ...')\n"
           "        subprocess.run(['rm', '-rf', AL_REPO_DIR], check=True)\n"
           "    print('Cloning CoFiPruning ...')\n"
           '    subprocess.run(\n'
           "        ['git', 'clone', 'https://github.com/princeton-nlp/CoFiPruning.git', AL_REPO_DIR],\n"
           '        check=True)\n'
           "    print('Clone done.\\n')\n"
           '\n'
           '    # ── Create modeling_albert.py in models/ ───────────────────────────────────\n'
           '    # IMPORTANT: Replace the placeholder below with the full contents of your\n'
           '    # already-patched modeling_albert.py before running block 0.\n'
           '    modeling_albert_code = r"""\n'
           '# CoFi-adapted ALBERT for sequence classification.\n'
           '\n'
           '# Architecture notes (albert-base-v2):\n'
           '#   - embedding_size=128, hidden_size=768 (factorised embedding)\n'
           '#   - embedding_hidden_mapping_in: Linear(128, 768) — projection inside AlbertTransformer\n'
           '#   - num_hidden_groups=1, inner_group_num=1 → ONE physical AlbertLayer shared across all\n'
           '#     num_hidden_layers=12 forward passes (parameter sharing)\n'
           '#   - AlbertAttention attrs: query, key, value, dense, LayerNorm (no .self / .output sub-objects)\n'
           '#   - AlbertLayer attrs: attention, ffn (up), ffn_output (down), full_layer_layer_norm, activation\n'
           '#   - AlbertModel.pooler: plain nn.Linear(hidden_size, hidden_size) — not a sub-module with .dense\n'
           '#   - AlbertForSequenceClassification: dropout + classifier (no pre_classifier)\n'
           '\n'
           '# CoFi masking strategy (option b — 12 logical masks, 1 physical layer):\n'
           '#   head_z         [num_hidden_layers, num_heads]     per-head mask per logical call\n'
           '#   head_layer_z   [num_hidden_layers]                per-call attention output gate\n'
           '#   intermediate_z [num_hidden_layers, intermediate]  per-neuron FFN mask per logical call\n'
           '#   mlp_z          [num_hidden_layers]                per-call FFN output gate\n'
           '#   hidden_z       [hidden_size]                      global hidden dim mask (applied once)\n'
           '\n'
           '#   All 12 logical masks train against the same physical weights — the optimizer\n'
           '#   receives 12 gradient signals per step. Mathematically impure but practically\n'
           '#   workable and requires no changes to L0Module or run_glue_prune.py.\n'
           '\n'
           '# Zero-head/zero-dim guard: if head_z zeroes all heads OR value/dense are None after\n'
           '# pruning, attention forward returns the residual unchanged. Same for FFN.\n'
           '\n'
           '\n'
           'import math\n'
           'import logging\n'
           'from typing import Optional, Tuple\n'
           '\n'
           'import torch\n'
           'import torch.nn as nn\n'
           'import torch.nn.functional as F\n'
           'from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss\n'
           '\n'
           'from transformers.models.albert.modeling_albert import (\n'
           '    AlbertPreTrainedModel,\n'
           '    AlbertEmbeddings,\n'
           '    AlbertModel,\n'
           '    AlbertForSequenceClassification,\n'
           ')\n'
           'from transformers.modeling_outputs import (\n'
           '    BaseModelOutputWithPooling,\n'
           '    SequenceClassifierOutput,\n'
           ')\n'
           'from transformers.modeling_utils import prune_linear_layer, find_pruneable_heads_and_indices\n'
           '\n'
           'logger = logging.getLogger(__name__)\n'
           '\n'
           '\n'
           '# ── CoFiLayerNorm ──────────────────────────────────────────────────────────────\n'
           'class CoFiLayerNorm(nn.LayerNorm):\n'
           '    # LayerNorm that accepts an optional hidden_z mask (same pattern as modeling_bert.py).\n'
           '\n'
           '    def forward(self, input, hidden_z=None):\n'
           '        if hidden_z is not None:\n'
           '            remaining = torch.where(~hidden_z.eq(0))[0]\n'
           '            compressed = torch.index_select(input, dim=-1, index=remaining)\n'
           '            normed = F.layer_norm(\n'
           '                compressed, [len(remaining)],\n'
           '                self.weight[remaining], self.bias[remaining], self.eps)\n'
           '            out = input.clone()\n'
           '            out[:, :, remaining] = normed\n'
           '            return out\n'
           '        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)\n'
           '\n'
           '\n'
           '# ── CoFiAlbertAttention ────────────────────────────────────────────────────────\n'
           'class CoFiAlbertAttention(nn.Module):\n'
           '\n'
           '    # AlbertAttention with CoFi head_z / head_layer_z / hidden_z masking.\n'
           '\n'
           "    # Albert's attention is self-contained: query/key/value/dense/LayerNorm are\n"
           '    # direct attrs (no .self / .output sub-objects). The post-attention residual\n'
           '    # add + LayerNorm is done INSIDE this module, same as the original.\n'
           '\n'
           '    # Zero-head guard: if all heads are pruned (value is None or n_heads==0),\n'
           '    # return hidden_states unchanged so the residual path stays intact.\n'
           '\n'
           '    def __init__(self, config):\n'
           '        super().__init__()\n'
           '        self.num_attention_heads = config.num_attention_heads\n'
           '        self.hidden_size = config.hidden_size\n'
           '        self.attention_head_size = config.hidden_size // config.num_attention_heads\n'
           '        self.all_head_size = self.num_attention_heads * self.attention_head_size\n'
           '        self.pruned_heads = set()\n'
           '\n'
           '        self.query = nn.Linear(config.hidden_size, self.all_head_size)\n'
           '        self.key   = nn.Linear(config.hidden_size, self.all_head_size)\n'
           '        self.value = nn.Linear(config.hidden_size, self.all_head_size)\n'
           '        self.dense = nn.Linear(config.hidden_size, config.hidden_size)\n'
           '\n'
           '        self.LayerNorm      = CoFiLayerNorm(config.hidden_size, eps=config.layer_norm_eps)\n'
           '        self.attn_dropout   = nn.Dropout(config.attention_probs_dropout_prob)\n'
           '        self.out_dropout    = nn.Dropout(config.hidden_dropout_prob)\n'
           '\n'
           '    def transpose_for_scores(self, x):\n'
           '        bs, seq, _ = x.size()\n'
           '        return x.view(bs, seq, self.num_attention_heads,\n'
           '                       self.attention_head_size).transpose(1, 2)\n'
           '\n'
           '    def forward(\n'
           '        self,\n'
           '        hidden_states,\n'
           '        attention_mask=None,\n'
           '        head_z=None,\n'
           '        head_layer_z=None,\n'
           '        hidden_z=None,\n'
           '    ):\n'
           '        # Zero-head guard\n'
           '        if self.value is None or self.num_attention_heads == 0:\n'
           '            return hidden_states  # pure residual — unchanged\n'
           '\n'
           '        q = self.transpose_for_scores(self.query(hidden_states))\n'
           '        k = self.transpose_for_scores(self.key(hidden_states))\n'
           '        v = self.transpose_for_scores(self.value(hidden_states))\n'
           '\n'
           '        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)\n'
           '        if attention_mask is not None:\n'
           '            scores = scores + attention_mask\n'
           '\n'
           '        probs   = F.softmax(scores, dim=-1)\n'
           '        probs   = self.attn_dropout(probs)\n'
           '        context = torch.matmul(probs, v)   # [bs, heads, seq, head_dim]\n'
           '\n'
           '        # head_z: per-head gate  [num_heads] or [1, num_heads, 1, 1]\n'
           '        if head_z is not None:\n'
           '            context = context * head_z.view(1, self.num_attention_heads, 1, 1)\n'
           '\n'
           '        bs, _, seq, _ = context.size()\n'
           '        context = context.transpose(1, 2).contiguous().view(\n'
           '            bs, seq, self.num_attention_heads * self.attention_head_size)\n'
           '\n'
           '        attn_out = self.dense(context)\n'
           '        attn_out = self.out_dropout(attn_out)\n'
           '\n'
           '        # head_layer_z: gate entire attention output before residual\n'
           '        if head_layer_z is not None:\n'
           '            attn_out = attn_out * head_layer_z\n'
           '\n'
           '        # Post-attention residual + LayerNorm (with hidden_z)\n'
           '        if hidden_z is not None:\n'
           '            attn_out = attn_out.mul(hidden_z)\n'
           '        normed = self.LayerNorm(attn_out + hidden_states, hidden_z)\n'
           '        if hidden_z is not None:\n'
           '            normed = normed.mul(hidden_z)\n'
           '\n'
           '        return normed\n'
           '\n'
           '    def prune_heads(self, heads):\n'
           '        if not heads:\n'
           '            return\n'
           '        heads, index = find_pruneable_heads_and_indices(\n'
           '            heads, self.num_attention_heads,\n'
           '            self.attention_head_size, self.pruned_heads)\n'
           '        if len(index) == 0:\n'
           '            self.query = self.key = self.value = self.dense = None\n'
           '            self.num_attention_heads = 0\n'
           '        else:\n'
           '            self.query = prune_linear_layer(self.query, index)\n'
           '            self.key   = prune_linear_layer(self.key,   index)\n'
           '            self.value = prune_linear_layer(self.value, index)\n'
           '            self.dense = prune_linear_layer(self.dense, index, dim=1)\n'
           '            self.num_attention_heads -= len(heads)\n'
           '            self.all_head_size = self.attention_head_size * self.num_attention_heads\n'
           '        self.pruned_heads |= heads\n'
           '\n'
           '\n'
           '# ── CoFiAlbertLayer ────────────────────────────────────────────────────────────\n'
           'class CoFiAlbertLayer(nn.Module):\n'
           '    # AlbertLayer with full CoFi mask support.\n'
           '\n'
           '    # Albert layer structure:\n'
           '    #   attention → ffn (up Linear) → activation → ffn_output (down Linear)\n'
           '    #   → full_layer_layer_norm(residual + ffn_out)\n'
           '\n'
           '    # NO shared-tensor issue: ffn and ffn_output are independent Linear attrs,\n'
           '    # not extracted references from a sub-module, so safetensors save works fine.\n'
           '\n'
           '    # Zero-FFN guard: if ffn is None (whole FFN pruned), skip FFN and apply\n'
           '    # full_layer_layer_norm on the attention output directly.\n'
           '\n'
           '    def __init__(self, config):\n'
           '        super().__init__()\n'
           '        self.attention            = CoFiAlbertAttention(config)\n'
           '        self.ffn                  = nn.Linear(config.hidden_size, config.intermediate_size)\n'
           '        self.ffn_output           = nn.Linear(config.intermediate_size, config.hidden_size)\n'
           '        self.full_layer_layer_norm = CoFiLayerNorm(config.hidden_size, eps=config.layer_norm_eps)\n'
           '        self.activation           = nn.GELU()\n'
           '        self.dropout              = nn.Dropout(config.hidden_dropout_prob)\n'
           '\n'
           '    def prune_heads(self, heads):\n'
           '        self.attention.prune_heads(heads)\n'
           '\n'
           '    def forward(\n'
           '        self,\n'
           '        hidden_states,\n'
           '        attention_mask=None,\n'
           '        head_z=None,\n'
           '        head_layer_z=None,\n'
           '        intermediate_z=None,\n'
           '        mlp_z=None,\n'
           '        hidden_z=None,\n'
           '    ):\n'
           '        attn_out = self.attention(\n'
           '            hidden_states,\n'
           '            attention_mask=attention_mask,\n'
           '            head_z=head_z,\n'
           '            head_layer_z=head_layer_z,\n'
           '            hidden_z=hidden_z,\n'
           '        )\n'
           '\n'
           '        if self.ffn is None:\n'
           '            # FFN fully pruned — apply post-FFN LayerNorm on attention output\n'
           '            if hidden_z is not None:\n'
           '                attn_out = attn_out.mul(hidden_z)\n'
           '            out = self.full_layer_layer_norm(attn_out, hidden_z)\n'
           '            if hidden_z is not None:\n'
           '                out = out.mul(hidden_z)\n'
           '            return out\n'
           '\n'
           '        # Up-projection + activation\n'
           '        inter = self.activation(self.ffn(attn_out))\n'
           '\n'
           '        # intermediate_z: per-neuron FFN mask\n'
           '        if intermediate_z is not None:\n'
           '            inter = inter.mul(intermediate_z)\n'
           '\n'
           '        # Down-projection\n'
           '        ffn_out = self.ffn_output(inter)\n'
           '        ffn_out = self.dropout(ffn_out)\n'
           '\n'
           '        # mlp_z: gate entire FFN output before residual\n'
           '        if mlp_z is not None:\n'
           '            ffn_out = ffn_out * mlp_z\n'
           '\n'
           '        # Post-FFN residual + LayerNorm\n'
           '        if hidden_z is not None:\n'
           '            ffn_out = ffn_out.mul(hidden_z)\n'
           '        out = self.full_layer_layer_norm(ffn_out + attn_out, hidden_z)\n'
           '        if hidden_z is not None:\n'
           '            out = out.mul(hidden_z)\n'
           '        return out\n'
           '\n'
           '\n'
           '# ── CoFiAlbertLayerGroup ───────────────────────────────────────────────────────\n'
           'class CoFiAlbertLayerGroup(nn.Module):\n'
           '    # Thin wrapper — passes CoFi masks through to CoFiAlbertLayer.\n'
           '\n'
           '    def __init__(self, config):\n'
           '        super().__init__()\n'
           '        self.albert_layers = nn.ModuleList(\n'
           '            [CoFiAlbertLayer(config) for _ in range(config.inner_group_num)])\n'
           '\n'
           '    def forward(self, hidden_states, attention_mask=None,\n'
           '                head_z=None, head_layer_z=None,\n'
           '                intermediate_z=None, mlp_z=None, hidden_z=None,\n'
           '                output_hidden_states=False):\n'
           '        all_hidden = ()\n'
           '        for lyr in self.albert_layers:\n'
           '            hidden_states = lyr(\n'
           '                hidden_states,\n'
           '                attention_mask=attention_mask,\n'
           '                head_z=head_z,\n'
           '                head_layer_z=head_layer_z,\n'
           '                intermediate_z=intermediate_z,\n'
           '                mlp_z=mlp_z,\n'
           '                hidden_z=hidden_z,\n'
           '            )\n'
           '            if output_hidden_states:\n'
           '                all_hidden = all_hidden + (hidden_states,)\n'
           '        return (hidden_states, all_hidden)\n'
           '\n'
           '\n'
           '# ── CoFiAlbertTransformer ──────────────────────────────────────────────────────\n'
           'class CoFiAlbertTransformer(nn.Module):\n'
           '    \n'
           '    # AlbertTransformer that threads CoFi masks per logical layer call.\n'
           '\n'
           '    # Parameter sharing: num_hidden_groups=1 → 1 physical CoFiAlbertLayerGroup,\n'
           '    # called num_hidden_layers=12 times. Each call receives the mask slice for\n'
           '    # its logical layer index i (head_z[i], intermediate_z[i], etc.).\n'
           '\n'
           '    # hidden_z is applied ONCE after embedding_hidden_mapping_in, then passed\n'
           '    # through to each layer call for post-residual masking.\n'
           '    \n'
           '\n'
           '    def __init__(self, config):\n'
           '        super().__init__()\n'
           '        self.config = config\n'
           '        self.embedding_hidden_mapping_in = nn.Linear(\n'
           '            config.embedding_size, config.hidden_size)\n'
           '        self.albert_layer_groups = nn.ModuleList(\n'
           '            [CoFiAlbertLayerGroup(config)\n'
           '             for _ in range(config.num_hidden_groups)])\n'
           '\n'
           '    def forward(self, hidden_states, attention_mask=None,\n'
           '                head_z=None, head_layer_z=None,\n'
           '                intermediate_z=None, mlp_z=None, hidden_z=None,\n'
           '                output_hidden_states=False):\n'
           '\n'
           '        hidden_states = self.embedding_hidden_mapping_in(hidden_states)\n'
           '\n'
           '        # Apply hidden_z after projection (embedding_size→hidden_size)\n'
           '        if hidden_z is not None:\n'
           '            hidden_states = hidden_states.mul(hidden_z)\n'
           '\n'
           '        all_hidden = (hidden_states,) if output_hidden_states else None\n'
           '\n'
           '        for i in range(self.config.num_hidden_layers):\n'
           '            group_idx = int(\n'
           '                i / (self.config.num_hidden_layers / self.config.num_hidden_groups))\n'
           '            grp_out = self.albert_layer_groups[group_idx](\n'
           '                hidden_states,\n'
           '                attention_mask=attention_mask,\n'
           '                # Option (a): single shared mask — index 0 for all 12 logical layer calls\n'
           '                head_z=head_z[0]         if head_z         is not None else None,\n'
           '                head_layer_z=head_layer_z[0] if head_layer_z is not None else None,\n'
           '                intermediate_z=intermediate_z[0] if intermediate_z is not None else None,\n'
           '                mlp_z=mlp_z[0]           if mlp_z           is not None else None,\n'
           '                hidden_z=hidden_z,\n'
           '                output_hidden_states=output_hidden_states,\n'
           '            )\n'
           '            hidden_states = grp_out[0]\n'
           '            if output_hidden_states:\n'
           '                all_hidden = all_hidden + grp_out[1]\n'
           '\n'
           '        return hidden_states, all_hidden\n'
           '\n'
           '\n'
           '# ── CoFiAlbertModel ────────────────────────────────────────────────────────────\n'
           'class CoFiAlbertModel(nn.Module):\n'
           '    #AlbertModel with CoFi masks threaded through encoder and embeddings.\n'
           '\n'
           '    def __init__(self, config):\n'
           '        super().__init__()\n'
           '        self.config     = config\n'
           '        self.embeddings = AlbertEmbeddings(config)  # standard, no CoFi changes needed\n'
           '        self.encoder    = CoFiAlbertTransformer(config)\n'
           '        self.pooler     = nn.Linear(config.hidden_size, config.hidden_size)\n'
           '        self.pooler_activation = nn.Tanh()\n'
           '\n'
           '    def _prune_heads(self, heads_to_prune):\n'
           '        # Prune heads — heads_to_prune: dict of {layer_num: list of heads to prune}.\n'
           '        # For ALBERT (shared layer), all layer entries map to the same physical layer.\n'
           '        # We take the union of all heads to prune across logical layers.\n'
           '        \n'
           '        all_heads = set()\n'
           '        for heads in heads_to_prune.values():\n'
           '            all_heads.update(heads)\n'
           '        if all_heads:\n'
           '            self.encoder.albert_layer_groups[0].albert_layers[0].prune_heads(all_heads)\n'
           '\n'
           '    def get_input_embeddings(self):\n'
           '        return self.embeddings.word_embeddings\n'
           '\n'
           '    def set_input_embeddings(self, value):\n'
           '        self.embeddings.word_embeddings = value\n'
           '\n'
           '    def get_extended_attention_mask(self, attention_mask, input_shape):\n'
           '        if attention_mask.dim() == 2:\n'
           '            extended = attention_mask[:, None, None, :]\n'
           '            extended = (1.0 - extended) * torch.finfo(torch.float32).min\n'
           '            return extended\n'
           '        return attention_mask\n'
           '\n'
           '    def forward(\n'
           '        self,\n'
           '        input_ids=None,\n'
           '        attention_mask=None,\n'
           '        token_type_ids=None,\n'
           '        position_ids=None,\n'
           '        inputs_embeds=None,\n'
           '        output_hidden_states=None,\n'
           '        head_z=None,\n'
           '        head_layer_z=None,\n'
           '        intermediate_z=None,\n'
           '        mlp_z=None,\n'
           '        hidden_z=None,\n'
           '    ):\n'
           '        output_hidden_states = output_hidden_states or False\n'
           '\n'
           '        if input_ids is not None and inputs_embeds is not None:\n'
           "            raise ValueError('Specify either input_ids or inputs_embeds, not both')\n"
           '        elif input_ids is not None:\n'
           '            input_shape = input_ids.size()\n'
           '            device = input_ids.device\n'
           '        elif inputs_embeds is not None:\n'
           '            input_shape = inputs_embeds.size()[:-1]\n'
           '            device = inputs_embeds.device\n'
           '        else:\n'
           "            raise ValueError('Specify input_ids or inputs_embeds')\n"
           '\n'
           '        if attention_mask is None:\n'
           '            attention_mask = torch.ones(input_shape, device=device)\n'
           '        if token_type_ids is None:\n'
           '            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)\n'
           '\n'
           '        extended_mask = self.get_extended_attention_mask(attention_mask, input_shape)\n'
           '\n'
           '        embedding_output = self.embeddings(\n'
           '            input_ids=input_ids,\n'
           '            token_type_ids=token_type_ids,\n'
           '            position_ids=position_ids,\n'
           '            inputs_embeds=inputs_embeds,\n'
           '        )\n'
           '\n'
           '        sequence_output, all_hidden = self.encoder(\n'
           '            embedding_output,\n'
           '            attention_mask=extended_mask,\n'
           '            head_z=head_z,\n'
           '            head_layer_z=head_layer_z,\n'
           '            intermediate_z=intermediate_z,\n'
           '            mlp_z=mlp_z,\n'
           '            hidden_z=hidden_z,\n'
           '            output_hidden_states=output_hidden_states,\n'
           '        )\n'
           '\n'
           '        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))\n'
           '\n'
           '        return sequence_output, pooled_output, all_hidden\n'
           '\n'
           '\n'
           '# ── CoFiAlbertForSequenceClassification ───────────────────────────────────────\n'
           'class CoFiAlbertForSequenceClassification(AlbertPreTrainedModel):\n'
           '    \n'
           '    # CoFi-prunable ALBERT for sequence classification.\n'
           '    # Mirrors CoFiBertForSequenceClassification from modeling_bert.py.\n'
           '    \n'
           '\n'
           '    def __init__(self, config):\n'
           '        super().__init__(config)\n'
           '        self.num_labels = config.num_labels\n'
           '        self.albert     = CoFiAlbertModel(config)\n'
           '        self.dropout    = nn.Dropout(config.classifier_dropout_prob)\n'
           '        self.classifier = nn.Linear(config.hidden_size, config.num_labels)\n'
           '\n'
           "        self.do_layer_distill = getattr(config, 'do_layer_distill', False)\n"
           '        self.layer_transformation = (\n'
           '            nn.Linear(config.hidden_size, config.hidden_size)\n'
           '            if self.do_layer_distill else None)\n'
           '\n'
           '        self.post_init()\n'
           '\n'
           '    def forward(\n'
           '        self,\n'
           '        input_ids=None,\n'
           '        attention_mask=None,\n'
           '        token_type_ids=None,\n'
           '        position_ids=None,\n'
           '        inputs_embeds=None,\n'
           '        labels=None,\n'
           '        output_attentions=None,   # accepted for API compat, ignored\n'
           '        output_hidden_states=None,\n'
           '        return_dict=None,\n'
           '        head_z=None,\n'
           '        head_layer_z=None,\n'
           '        intermediate_z=None,\n'
           '        mlp_z=None,\n'
           '        hidden_z=None,\n'
           '    ):\n'
           '        sequence_output, pooled_output, all_hidden = self.albert(\n'
           '            input_ids=input_ids,\n'
           '            attention_mask=attention_mask,\n'
           '            token_type_ids=token_type_ids,\n'
           '            position_ids=position_ids,\n'
           '            inputs_embeds=inputs_embeds,\n'
           '            output_hidden_states=True if self.training else output_hidden_states,\n'
           '            head_z=head_z,\n'
           '            head_layer_z=head_layer_z,\n'
           '            intermediate_z=intermediate_z,\n'
           '            mlp_z=mlp_z,\n'
           '            hidden_z=hidden_z,\n'
           '        )\n'
           '\n'
           '        pooled_output = self.dropout(pooled_output)\n'
           '        logits        = self.classifier(pooled_output)\n'
           '\n'
           '        loss = None\n'
           '        if labels is not None:\n'
           '            if self.config.problem_type is None:\n'
           '                if self.num_labels == 1:\n'
           "                    self.config.problem_type = 'regression'\n"
           '                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):\n'
           "                    self.config.problem_type = 'single_label_classification'\n"
           '                else:\n'
           "                    self.config.problem_type = 'multi_label_classification'\n"
           '\n'
           "            if self.config.problem_type == 'regression':\n"
           '                loss_fct = MSELoss()\n'
           '                loss = (loss_fct(logits.squeeze(), labels.squeeze())\n'
           '                        if self.num_labels == 1 else loss_fct(logits, labels))\n'
           "            elif self.config.problem_type == 'single_label_classification':\n"
           '                loss_fct = CrossEntropyLoss()\n'
           '                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))\n'
           "            elif self.config.problem_type == 'multi_label_classification':\n"
           '                loss_fct = BCEWithLogitsLoss()\n'
           '                loss = loss_fct(logits, labels)\n'
           '\n'
           '        # Pack outputs to match what CoFiTrainer expects:\n'
           '        #   outputs[0] = loss (or logits if no loss)\n'
           '        #   outputs[1] = logits\n'
           '        #   outputs[2] = all_hidden_states tuple (for layer distillation)\n'
           '        if not return_dict:\n'
           '            output = (logits,) + ((all_hidden,) if all_hidden else ())\n'
           '            return ((loss,) + output) if loss is not None else output\n'
           '\n'
           '        return SequenceClassifierOutput(\n'
           '            loss=loss,\n'
           '            logits=logits,\n'
           '            hidden_states=all_hidden,\n'
           '            attentions=None,\n'
           '        )"""\n'
           "    modeling_albert_path = os.path.join(AL_REPO_DIR, 'models', 'modeling_albert.py')\n"
           "    with open(modeling_albert_path, 'w') as f:\n"
           '        f.write(modeling_albert_code.strip())\n'
           "    print('  [created]         models/modeling_albert.py')\n"
           '\n'
           '    # ── Standard import patches (all .py files) ────────────────────────────────\n'
           "    print('\\n  Standard import patches...')\n"
           '    IMPORT_PATCHES = [\n'
           "        ('from transformers.file_utils import hf_bucket_url, cached_path',\n"
           "         'from huggingface_hub import cached_download as cached_path'),\n"
           "        ('from transformers.file_utils import cached_path',\n"
           "         'from huggingface_hub import cached_download as cached_path'),\n"
           "        ('from transformers.file_utils import hf_bucket_url', ''),\n"
           "        ('from datasets import load_dataset, load_metric, DatasetDict',\n"
           "         'from datasets import load_dataset, DatasetDict\\nimport evaluate'),\n"
           '        (\'metric = load_metric("glue", data_args.task_name)\',\n'
           '         \'metric = evaluate.load("glue", data_args.task_name)\'),\n'
           '        (\'metric = load_metric("accuracy")\',\n'
           '         \'metric = evaluate.load("accuracy")\'),\n'
           "        ('from black import main', ''),\n"
           '    ]\n'
           '    for root, dirs, files in os.walk(AL_REPO_DIR):\n'
           "        dirs[:] = [d for d in dirs if d != '.git']\n"
           '        for fname in files:\n'
           "            if not fname.endswith('.py'):\n"
           '                continue\n'
           '            fpath = os.path.join(root, fname)\n'
           '            rel   = os.path.relpath(fpath, AL_REPO_DIR)\n'
           '            for old, new in IMPORT_PATCHES:\n'
           '                patch_file(fpath, old, new, rel)\n'
           '\n'
           '    # ── run_glue_prune.py ──────────────────────────────────────────────────────\n'
           "    print('\\n  Patching run_glue_prune.py...')\n"
           "    glue_path = os.path.join(AL_REPO_DIR, 'run_glue_prune.py')\n"
           '\n'
           '    patch_file(glue_path,\n'
           '        \'load_dataset("glue", data_args.task_name)\',\n'
           '        \'load_dataset("glue", data_args.task_name, trust_remote_code=True)\',\n'
           "        'trust_remote_code')\n"
           '\n'
           '    patch_file(glue_path,\n'
           '        \'"evaluation_strategy"\',\n'
           '        \'"eval_strategy"\',\n'
           "        'evaluation_strategy -> eval_strategy')\n"
           '\n'
           '    patch_file(glue_path,\n'
           "        'from models.modeling_bert import CoFiBertForSequenceClassification\\n'\n"
           "        'from models.modeling_roberta import CoFiRobertaForSequenceClassification',\n"
           "        'from models.modeling_bert import CoFiBertForSequenceClassification\\n'\n"
           "        'from models.modeling_roberta import CoFiRobertaForSequenceClassification\\n'\n"
           "        'from models.modeling_albert import CoFiAlbertForSequenceClassification',\n"
           "        'import CoFiAlbertForSequenceClassification')\n"
           '\n'
           '    patch_file(glue_path,\n'
           "        '    Model = CoFiBertForSequenceClassification if model_args.model_name_or_path.startswith(\\n'\n"
           '        \'        "bert") else CoFiRobertaForSequenceClassification\',\n'
           "        '    _cfg_check = AutoConfig.from_pretrained(model_args.model_name_or_path)\\n'\n"
           '        \'    if _cfg_check.model_type == "albert":\\n\'\n'
           "        '        Model = CoFiAlbertForSequenceClassification\\n'\n"
           '        \'    elif _cfg_check.model_type in ("bert",):\\n\'\n'
           "        '        Model = CoFiBertForSequenceClassification\\n'\n"
           "        '    else:\\n'\n"
           "        '        Model = CoFiRobertaForSequenceClassification',\n"
           "        'model selection via config.model_type')\n"
           '\n'
           '    # ALBERT option (a): L0Module gets num_hidden_layers=num_hidden_groups (=1)\n'
           '    import re\n'
           '    with open(glue_path) as f:\n'
           '        src = f.read()\n'
           "    m = re.search(r'        l0_module = L0Module\\(config=config,.*?\\)', src, re.DOTALL)\n"
           '    if m:\n'
           '        full_inst = m.group()\n'
           '        new_inst = (\n'
           "            '        # ALBERT option (a): override num_hidden_layers to num_hidden_groups (=1)\\n'\n"
           "            '        # so L0Module initializes ONE shared mask set instead of 12 independent ones.\\n'\n"
           "            '        _l0_config = config\\n'\n"
           '            \'        if getattr(config, "model_type", "") == "albert":\\n\'\n'
           "            '            _l0_config = deepcopy(config)\\n'\n"
           '            \'            _l0_config.num_hidden_layers = getattr(config, "num_hidden_groups", 1)\\n\'\n'
           "            '    ' + full_inst.replace('config=config', 'config=_l0_config')\n"
           '        )\n'
           '        if new_inst not in src:\n'
           "            with open(glue_path, 'w') as f:\n"
           '                f.write(src.replace(full_inst, new_inst))\n'
           "            print('  [patched]         run_glue_prune.py: ALBERT L0Module uses num_hidden_groups=1')\n"
           '        else:\n'
           "            print('  [already patched] run_glue_prune.py: ALBERT L0Module')\n"
           '    else:\n'
           "        print('  [not found]       run_glue_prune.py: L0Module instantiation')\n"
           '\n'
           '    # ── models/modeling_bert.py ────────────────────────────────────────────────\n'
           "    print('\\n  Patching models/modeling_bert.py...')\n"
           "    bert_path = os.path.join(AL_REPO_DIR, 'models', 'modeling_bert.py')\n"
           '    src = open(bert_path).read()\n'
           '    pat = re.compile(\n'
           "        r'[ \\t]*@classmethod\\s*\\n[ \\t]*def from_pretrained\\(cls.*?(?=\\n[ \\t]{0,4}(?:def |class "
           "|\\Z))',\n"
           '        re.DOTALL)\n'
           '    if re.search(pat, src):\n'
           "        open(bert_path, 'w').write(re.sub(pat, '', src))\n"
           "        print('  [patched]         removed from_pretrained override')\n"
           '    else:\n'
           "        print('  [already patched] no from_pretrained override found')\n"
           '\n'
           '    # ── utils/cofi_utils.py ────────────────────────────────────────────────────\n'
           "    print('\\n  Patching utils/cofi_utils.py...')\n"
           "    utils_path = os.path.join(AL_REPO_DIR, 'utils', 'cofi_utils.py')\n"
           '\n'
           '    patch_file(utils_path,\n'
           '        \'    p = os.path.join(model_path, "pytorch_model.bin")\\n\'\n'
           '        \'    loaded_weights = torch.load(p, map_location="cpu")\',\n'
           '        \'    p_bin  = os.path.join(model_path, "pytorch_model.bin")\\n\'\n'
           '        \'    p_safe = os.path.join(model_path, "model.safetensors")\\n\'\n'
           "        '    if os.path.exists(p_bin):\\n'\n"
           '        \'        loaded_weights = torch.load(p_bin, map_location="cpu")\\n\'\n'
           "        '    elif os.path.exists(p_safe):\\n'\n"
           "        '        from safetensors.torch import load_file\\n'\n"
           "        '        loaded_weights = load_file(p_safe)\\n'\n"
           "        '    else:\\n'\n"
           '        \'        raise FileNotFoundError(f"No weights found in {model_path}")\',\n'
           "        'cofi_utils: safetensors support')\n"
           '\n'
           '    patch_file(utils_path,\n'
           "        'def prune_model_with_z(zs, model):',\n"
           "        'def _get_layers(bert):\\n'\n"
           '        \'    """Return the list of transformer layer objects, model-agnostically."""\\n\'\n'
           '        \'    if hasattr(bert, "encoder"):\\n\'\n'
           "        '        enc = bert.encoder\\n'\n"
           '        \'        if hasattr(enc, "layer"):  # BERT / RoBERTa\\n\'\n'
           "        '            return enc.layer\\n'\n"
           '        \'        if hasattr(enc, "albert_layer_groups"):  # ALBERT\\n\'\n'
           "        '            return enc.albert_layer_groups[0].albert_layers\\n'\n"
           '        \'    if hasattr(bert, "transformer"):  # DistilBERT\\n\'\n'
           "        '        return bert.transformer.layer\\n'\n"
           '        \'    raise ValueError(f"Cannot find layers in {type(bert)}")\\n\\n\'\n'
           "        'def prune_model_with_z(zs, model):',\n"
           "        'cofi_utils: _get_layers helper')\n"
           '\n'
           '    patch_file(utils_path,\n'
           "        'def prune_model_with_z(zs, model):\\n'\n"
           "        '    if zs is None:\\n'\n"
           "        '        return None, None\\n'\n"
           '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\',\n'
           "        'def prune_model_with_z(zs, model):\\n'\n"
           "        '    if zs is None:\\n'\n"
           "        '        return None, None\\n'\n"
           '        \'    if hasattr(model, "bert"):\\n\'\n'
           "        '        bert = model.bert\\n'\n"
           "        '        num_layers = model.config.num_hidden_layers\\n'\n"
           '        \'    elif hasattr(model, "albert"):\\n\'\n'
           "        '        bert = model.albert\\n'\n"
           "        '        num_layers = model.config.num_hidden_layers\\n'\n"
           '        \'    elif hasattr(model, "roberta"):\\n\'\n'
           "        '        bert = model.roberta\\n'\n"
           "        '        num_layers = model.config.num_hidden_layers\\n'\n"
           '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
           "        '        bert = model.distilbert\\n'\n"
           "        '        num_layers = model.config.n_layers\\n'\n"
           "        '    else:\\n'\n"
           '        \'        raise ValueError(f"Unknown model type: {type(model)}")\',\n'
           "        'cofi_utils: model type detection in prune_model_with_z')\n"
           '\n'
           '    patch_file(utils_path,\n'
           "        'def update_params(model, zs):\\n'\n"
           '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\\n\'\n'
           "        '\\n'\n"
           "        '    config = model.config\\n'\n"
           "        '    hidden_dims = config.hidden_size\\n'\n"
           "        '    num_heads = config.num_attention_heads\\n'\n"
           "        '    dims_per_head = hidden_dims // num_heads\\n'\n"
           "        '    num_layers = config.num_hidden_layers',\n"
           "        'def update_params(model, zs):\\n'\n"
           '        \'    if hasattr(model, "bert"):\\n\'\n'
           "        '        bert = model.bert\\n'\n"
           "        '        num_layers = model.config.num_hidden_layers\\n'\n"
           '        \'    elif hasattr(model, "albert"):\\n\'\n'
           "        '        bert = model.albert\\n'\n"
           "        '        num_layers = model.config.num_hidden_layers\\n'\n"
           '        \'    elif hasattr(model, "roberta"):\\n\'\n'
           "        '        bert = model.roberta\\n'\n"
           "        '        num_layers = model.config.num_hidden_layers\\n'\n"
           '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
           "        '        bert = model.distilbert\\n'\n"
           "        '        num_layers = model.config.n_layers\\n'\n"
           "        '    else:\\n'\n"
           '        \'        raise ValueError(f"Unknown model type: {type(model)}")\\n\'\n'
           "        '\\n'\n"
           "        '    config = model.config\\n'\n"
           "        '    hidden_dims = config.hidden_size\\n'\n"
           "        '    num_heads = config.num_attention_heads\\n'\n"
           "        '    dims_per_head = hidden_dims // num_heads',\n"
           "        'cofi_utils: update_params model dispatch')\n"
           '\n'
           '    patch_file(utils_path,\n'
           "        '            bert.embeddings.token_type_embeddings.weight.data = \\\\\\n'\n"
           "        '                bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)',\n"
           '        \'            if hasattr(bert.embeddings, "token_type_embeddings"):\\n\'\n'
           "        '                bert.embeddings.token_type_embeddings.weight.data = \\\\\\n'\n"
           "        '                    bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)',\n"
           "        'cofi_utils: guard token_type_embeddings in update_params')\n"
           '\n'
           '    patch_file(utils_path,\n'
           '        \'        if "hidden_z" in zs:\\n\'\n'
           '        \'            hidden_z = zs["hidden_z"].cpu().squeeze().clone()\\n\'\n'
           "        '            bert.embeddings.word_embeddings.weight.data =\\\\\\n'\n"
           "        '                bert.embeddings.word_embeddings.weight.data.mul(hidden_z)\\n'\n"
           "        '            bert.embeddings.position_embeddings.weight.data = \\\\\\n'\n"
           "        '                bert.embeddings.position_embeddings.weight.data.mul(hidden_z)\\n'\n"
           '        \'            if hasattr(bert.embeddings, "token_type_embeddings"):\\n\'\n'
           "        '                bert.embeddings.token_type_embeddings.weight.data = \\\\\\n'\n"
           "        '                    bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)',\n"
           '        \'        if "hidden_z" in zs:\\n\'\n'
           '        \'            hidden_z = zs["hidden_z"].cpu().squeeze().clone()\\n\'\n'
           "        '            _emb_dim = bert.embeddings.word_embeddings.weight.shape[1]\\n'\n"
           "        '            if _emb_dim == hidden_z.shape[0]:\\n'\n"
           "        '                bert.embeddings.word_embeddings.weight.data = \\\\\\n'\n"
           "        '                    bert.embeddings.word_embeddings.weight.data.mul(hidden_z)\\n'\n"
           "        '                bert.embeddings.position_embeddings.weight.data = \\\\\\n'\n"
           "        '                    bert.embeddings.position_embeddings.weight.data.mul(hidden_z)\\n'\n"
           '        \'                if hasattr(bert.embeddings, "token_type_embeddings"):\\n\'\n'
           "        '                    bert.embeddings.token_type_embeddings.weight.data = \\\\\\n'\n"
           "        '                        bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)',\n"
           "        'cofi_utils: guard embedding hidden_z for ALBERT in update_params')\n"
           '\n'
           '    patch_file(utils_path,\n'
           "        '            for layer in range(num_layers):\\n'\n"
           "        '                bert.encoder.layer[layer].attention.self.key.weight.data = "
           "bert.encoder.layer[layer].attention.self.key.weight.data.mul(hidden_z)\\n'\n"
           "        '                bert.encoder.layer[layer].attention.self.query.weight.data = "
           "bert.encoder.layer[layer].attention.self.query.weight.data.mul(hidden_z)\\n'\n"
           "        '                bert.encoder.layer[layer].attention.self.value.weight.data = "
           "bert.encoder.layer[layer].attention.self.value.weight.data.mul(hidden_z)\\n'\n"
           "        '                bert.encoder.layer[layer].attention.output.dense.weight.data = "
           'bert.encoder.layer[layer].attention.output.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, '
           "1)\\n'\n"
           "        '                bert.encoder.layer[layer].attention.output.dense.bias.data = "
           "bert.encoder.layer[layer].attention.output.dense.bias.data.mul(hidden_z)\\n'\n"
           "        '                bert.encoder.layer[layer].intermediate.dense.weight.data = "
           "bert.encoder.layer[layer].intermediate.dense.weight.data.mul(hidden_z)\\n'\n"
           "        '                bert.encoder.layer[layer].output.dense.weight.data = "
           "bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)',\n"
           "        '            layers = _get_layers(bert)\\n'\n"
           "        '            for layer in range(len(layers)):\\n'\n"
           "        '                lyr = layers[layer]\\n'\n"
           '        \'                if hasattr(lyr, "attention") and hasattr(lyr.attention, "self"):\\n\'\n'
           "        '                    # BERT-style\\n'\n"
           "        '                    lyr.attention.self.key.weight.data = "
           "lyr.attention.self.key.weight.data.mul(hidden_z)\\n'\n"
           "        '                    lyr.attention.self.query.weight.data = "
           "lyr.attention.self.query.weight.data.mul(hidden_z)\\n'\n"
           "        '                    lyr.attention.self.value.weight.data = "
           "lyr.attention.self.value.weight.data.mul(hidden_z)\\n'\n"
           "        '                    lyr.attention.output.dense.weight.data = "
           "lyr.attention.output.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)\\n'\n"
           "        '                    lyr.attention.output.dense.bias.data = "
           "lyr.attention.output.dense.bias.data.mul(hidden_z)\\n'\n"
           "        '                    lyr.intermediate.dense.weight.data = "
           "lyr.intermediate.dense.weight.data.mul(hidden_z)\\n'\n"
           "        '                    lyr.output.dense.weight.data = lyr.output.dense.weight.data.transpose(0, "
           "1).mul(hidden_z).transpose(0, 1)\\n'\n"
           '        \'                elif hasattr(lyr, "attention") and hasattr(lyr.attention, "query") and '
           'hasattr(lyr, "ffn"):\\n\'\n'
           "        '                    # ALBERT-style\\n'\n"
           "        '                    if lyr.attention.key is not None:\\n'\n"
           "        '                        lyr.attention.key.weight.data = "
           "lyr.attention.key.weight.data.mul(hidden_z)\\n'\n"
           "        '                        lyr.attention.query.weight.data = "
           "lyr.attention.query.weight.data.mul(hidden_z)\\n'\n"
           "        '                        lyr.attention.value.weight.data = "
           "lyr.attention.value.weight.data.mul(hidden_z)\\n'\n"
           "        '                        lyr.attention.dense.weight.data = "
           "lyr.attention.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)\\n'\n"
           "        '                        lyr.attention.dense.bias.data = "
           "lyr.attention.dense.bias.data.mul(hidden_z)\\n'\n"
           "        '                    if lyr.ffn is not None:\\n'\n"
           "        '                        lyr.ffn.weight.data = lyr.ffn.weight.data.mul(hidden_z)\\n'\n"
           "        '                        lyr.ffn_output.weight.data = lyr.ffn_output.weight.data.transpose(0, "
           "1).mul(hidden_z).transpose(0, 1)\\n'\n"
           '        \'                elif hasattr(lyr, "attention") and hasattr(lyr.attention, "q_lin"):\\n\'\n'
           "        '                    # DistilBERT-style\\n'\n"
           "        '                    if lyr.attention.q_lin is not None:\\n'\n"
           "        '                        lyr.attention.q_lin.weight.data = "
           "lyr.attention.q_lin.weight.data.mul(hidden_z)\\n'\n"
           "        '                        lyr.attention.k_lin.weight.data = "
           "lyr.attention.k_lin.weight.data.mul(hidden_z)\\n'\n"
           "        '                        lyr.attention.v_lin.weight.data = "
           "lyr.attention.v_lin.weight.data.mul(hidden_z)\\n'\n"
           "        '                        lyr.attention.out_lin.weight.data = "
           "lyr.attention.out_lin.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)\\n'\n"
           "        '                        lyr.attention.out_lin.bias.data = "
           "lyr.attention.out_lin.bias.data.mul(hidden_z)\\n'\n"
           '        \'                    if hasattr(lyr, "ffn_lin1") and lyr.ffn_lin1 is not None:\\n\'\n'
           "        '                        lyr.ffn_lin1.weight.data = lyr.ffn_lin1.weight.data.mul(hidden_z)\\n'\n"
           "        '                        lyr.ffn_lin2.weight.data = lyr.ffn_lin2.weight.data.transpose(0, "
           "1).mul(hidden_z).transpose(0, 1)',\n"
           "        'cofi_utils: update_params hidden_z loop model-agnostic')\n"
           '\n'
           '    patch_file(utils_path,\n'
           "        '            for layer in range(num_layers):\\n'\n"
           '        \'                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()\\n\'\n'
           "        '                bert.encoder.layer[layer].output.dense.weight.data = "
           "bert.encoder.layer[layer].output.dense.weight.data.mul(intermediate_z)\\n'\n"
           '        \'                if "mlp_z" in zs:\\n\'\n'
           '        \'                    mlp_z = zs["mlp_z"][layer].cpu()\\n\'\n'
           "        '                    bert.encoder.layer[layer].output.dense.weight.data = "
           "bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)\\n'\n"
           "        '                    bert.encoder.layer[layer].output.dense.bias.data = "
           "bert.encoder.layer[layer].output.dense.bias.data.mul(mlp_z)\\n'\n"
           "        '\\n'\n"
           '        \'        if "head_z" in zs:\\n\'\n'
           "        '            for layer in range(num_layers):\\n'\n"
           '        \'                head_z = zs["head_z"][layer].cpu().squeeze().clone()\\n\'\n'
           "        '                head_z = torch.repeat_interleave(head_z, dims_per_head)\\n'\n"
           "        '                bert.encoder.layer[layer].attention.self.value.weight.data = "
           'bert.encoder.layer[layer].attention.self.value.weight.transpose(0, 1).data.mul(head_z).transpose(0, '
           "1)\\n'\n"
           "        '                bert.encoder.layer[layer].attention.self.value.bias.data = "
           "bert.encoder.layer[layer].attention.self.value.bias.data.mul(head_z)\\n'\n"
           '        \'                if "head_layer_z" in zs:\\n\'\n'
           '        \'                    head_layer_z = zs["head_layer_z"][layer].cpu()\\n\'\n'
           "        '                    bert.encoder.layer[layer].attention.output.dense.weight.data = "
           "bert.encoder.layer[\\n'\n"
           "        '                        layer].attention.output.dense.weight.transpose(0, "
           "1).data.mul(head_layer_z).transpose(0, 1)\\n'\n"
           "        '                    bert.encoder.layer[layer].attention.output.dense.bias.data = "
           "bert.encoder.layer[\\n'\n"
           "        '                        layer].attention.output.dense.bias.data.mul(head_layer_z)',\n"
           "        '            layers = _get_layers(bert)\\n'\n"
           '        \'            _nz_i = len(zs["intermediate_z"])  # 1 for ALBERT, num_layers for BERT\\n\'\n'
           "        '            for layer in range(len(layers)):\\n'\n"
           '        \'                intermediate_z = zs["intermediate_z"][min(layer, '
           "_nz_i-1)].cpu().squeeze().clone()\\n'\n"
           "        '                lyr = layers[layer]\\n'\n"
           '        \'                down = lyr.ffn_output if hasattr(lyr, "ffn_output") else lyr.output.dense\\n\'\n'
           "        '                if down is not None:\\n'\n"
           "        '                    down.weight.data = down.weight.data.mul(intermediate_z)\\n'\n"
           '        \'                    if "mlp_z" in zs:\\n\'\n'
           '        \'                        mlp_z = zs["mlp_z"][min(layer, _nz_i-1)].cpu()\\n\'\n'
           "        '                        down.weight.data = down.weight.data.transpose(0, "
           "1).mul(mlp_z).transpose(0, 1)\\n'\n"
           "        '                        down.bias.data = down.bias.data.mul(mlp_z)\\n'\n"
           "        '\\n'\n"
           '        \'        if "head_z" in zs:\\n\'\n'
           "        '            layers = _get_layers(bert)\\n'\n"
           '        \'            _nz_h = len(zs["head_z"])  # 1 for ALBERT, num_layers for BERT\\n\'\n'
           "        '            for layer in range(len(layers)):\\n'\n"
           "        '                _zi = min(layer, _nz_h - 1)\\n'\n"
           '        \'                head_z = zs["head_z"][_zi].cpu().squeeze().clone()\\n\'\n'
           "        '                head_z = torch.repeat_interleave(head_z, dims_per_head)\\n'\n"
           "        '                lyr = layers[layer]\\n'\n"
           '        \'                if hasattr(lyr, "attention") and hasattr(lyr.attention, "self"):\\n\'\n'
           "        '                    v = lyr.attention.self.value\\n'\n"
           "        '                    o = lyr.attention.output.dense\\n'\n"
           "        '                else:\\n'\n"
           '        \'                    v = getattr(lyr.attention, "value", None)\\n\'\n'
           '        \'                    o = getattr(lyr.attention, "dense", None)\\n\'\n'
           "        '                if v is not None:\\n'\n"
           "        '                    v.weight.data = v.weight.transpose(0, 1).data.mul(head_z).transpose(0, "
           "1)\\n'\n"
           "        '                    v.bias.data = v.bias.data.mul(head_z)\\n'\n"
           '        \'                if "head_layer_z" in zs:\\n\'\n'
           '        \'                    head_layer_z = zs["head_layer_z"][_zi].cpu()\\n\'\n'
           "        '                    if o is not None:\\n'\n"
           "        '                        o.weight.data = o.weight.transpose(0, "
           "1).data.mul(head_layer_z).transpose(0, 1)\\n'\n"
           "        '                        o.bias.data = o.bias.data.mul(head_layer_z)',\n"
           "        'cofi_utils: update_params intermediate_z and head_z loops model-agnostic + option-a')\n"
           '\n'
           '    patch_file(utils_path,\n'
           '        \'        if hasattr(model, "classifier"):\\n\'\n'
           '        \'            if hasattr(model.classifier, "dense"):\\n\'\n'
           "        '                model.classifier.dense = prune_linear_layer(model.classifier.dense, index, "
           "dim=1)\\n'\n"
           '        \'        if hasattr(model, "cls"):\\n\'\n'
           '        \'            if hasattr(model.cls, "dense"):\\n\'\n'
           "        '                model.cls.dense = prune_linear_layer(model.classifier.dense, index, dim=1)\\n'\n"
           '        \'        if hasattr(bert.pooler, "dense"):\\n\'\n'
           "        '            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)\\n'\n"
           '        \'        if hasattr(model, "qa_outputs"):\\n\'\n'
           "        '            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)',\n"
           '        \'        if hasattr(model, "pre_classifier"):\\n\'\n'
           "        '            model.pre_classifier = prune_linear_layer(model.pre_classifier, index, dim=1)\\n'\n"
           "        '            model.pre_classifier = prune_linear_layer(model.pre_classifier, index, dim=0)\\n'\n"
           '        \'        if hasattr(model, "classifier"):\\n\'\n'
           '        \'            if hasattr(model.classifier, "dense"):\\n\'\n'
           "        '                model.classifier.dense = prune_linear_layer(model.classifier.dense, index, "
           "dim=1)\\n'\n"
           "        '            elif isinstance(model.classifier, torch.nn.Linear):\\n'\n"
           "        '                model.classifier = prune_linear_layer(model.classifier, index, dim=1)\\n'\n"
           '        \'        if hasattr(model, "cls"):\\n\'\n'
           '        \'            if hasattr(model.cls, "dense"):\\n\'\n'
           "        '                model.cls.dense = prune_linear_layer(model.classifier.dense, index, dim=1)\\n'\n"
           '        \'        if hasattr(bert, "pooler") and bert.pooler is not None:\\n\'\n'
           "        '            if isinstance(bert.pooler, torch.nn.Linear):\\n'\n"
           "        '                bert.pooler = prune_linear_layer(bert.pooler, index, dim=1)\\n'\n"
           "        '                bert.pooler = prune_linear_layer(bert.pooler, index, dim=0)\\n'\n"
           '        \'            elif hasattr(bert.pooler, "dense"):\\n\'\n'
           "        '                bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)\\n'\n"
           '        \'        if hasattr(model, "qa_outputs"):\\n\'\n'
           "        '            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)\\n'\n"
           '        \'        if getattr(model, "layer_transformation", None) is not None:\\n\'\n'
           "        '            if model.layer_transformation.weight.shape[1] == len(index):\\n'\n"
           "        '                model.layer_transformation = prune_linear_layer(model.layer_transformation, "
           "index, dim=1)\\n'\n"
           "        '                model.layer_transformation = prune_linear_layer(model.layer_transformation, "
           "index, dim=0)\\n'\n"
           '        \'            print("layer transformation", model.layer_transformation.weight.shape)\\n\'\n'
           '        \'        if getattr(model, "mha_layer_transformation", None) is not None:\\n\'\n'
           "        '            model.mha_layer_transformation = prune_linear_layer(model.mha_layer_transformation, "
           "index, dim=1)\\n'\n"
           '        \'            print("layer mha_layer_transformation", '
           "model.mha_layer_transformation.weight.shape)',\n"
           "        'cofi_utils: classifier/pooler/layer_transformation - albert + distilbert + bert')\n"
           '\n'
           '    patch_file(utils_path,\n'
           "        '        for layer in range(0, 12):\\n'\n"
           "        '            if bert.encoder.layer[layer].attention.self.query is not None:\\n'\n"
           "        '                bert.encoder.layer[layer].attention.self.query = \\\\\\n'\n"
           "        '                    prune_layer(bert.encoder.layer[layer].attention.self.query , index, "
           "dim=1)\\n'\n"
           "        '                bert.encoder.layer[layer].attention.self.key = \\\\\\n'\n"
           "        '                    prune_layer(bert.encoder.layer[layer].attention.self.key , index, dim=1)\\n'\n"
           "        '            if bert.encoder.layer[layer].attention.self.value is not None:\\n'\n"
           "        '                bert.encoder.layer[layer].attention.self.value = \\\\\\n'\n"
           "        '                    prune_layer(bert.encoder.layer[layer].attention.self.value , index, "
           "dim=1)\\n'\n"
           "        '                bert.encoder.layer[layer].attention.output.dense = \\\\\\n'\n"
           "        '                    prune_layer(bert.encoder.layer[layer].attention.output.dense , index, "
           "dim=0)\\n'\n"
           "        '                prune_layer_norm(bert.encoder.layer[layer].attention.output.LayerNorm, "
           "index)\\n'\n"
           "        '            if bert.encoder.layer[layer].intermediate.dense is not None:\\n'\n"
           "        '                bert.encoder.layer[layer].intermediate.dense = \\\\\\n'\n"
           "        '                    prune_layer( bert.encoder.layer[layer].intermediate.dense, index, dim=1)\\n'\n"
           "        '                bert.encoder.layer[layer].output.dense = \\\\\\n'\n"
           "        '                    prune_layer( bert.encoder.layer[layer].output.dense, index, dim=0)\\n'\n"
           "        '                prune_layer_norm(bert.encoder.layer[layer].output.LayerNorm, index)',\n"
           "        '        layers = _get_layers(bert)\\n'\n"
           "        '        for layer in range(len(layers)):\\n'\n"
           "        '            lyr = layers[layer]\\n'\n"
           '        \'            if hasattr(lyr, "attention") and hasattr(lyr.attention, "self"):\\n\'\n'
           "        '                # BERT-style\\n'\n"
           "        '                if lyr.attention.self.query is not None:\\n'\n"
           "        '                    lyr.attention.self.query = prune_layer(lyr.attention.self.query, index, "
           "dim=1)\\n'\n"
           "        '                    lyr.attention.self.key   = prune_layer(lyr.attention.self.key,   index, "
           "dim=1)\\n'\n"
           "        '                if lyr.attention.self.value is not None:\\n'\n"
           "        '                    lyr.attention.self.value = prune_layer(lyr.attention.self.value, index, "
           "dim=1)\\n'\n"
           "        '                    lyr.attention.output.dense = prune_layer(lyr.attention.output.dense, index, "
           "dim=0)\\n'\n"
           "        '                    prune_layer_norm(lyr.attention.output.LayerNorm, index)\\n'\n"
           "        '                if lyr.intermediate.dense is not None:\\n'\n"
           "        '                    lyr.intermediate.dense = prune_layer(lyr.intermediate.dense, index, "
           "dim=1)\\n'\n"
           "        '                    lyr.output.dense       = prune_layer(lyr.output.dense,       index, "
           "dim=0)\\n'\n"
           "        '                    prune_layer_norm(lyr.output.LayerNorm, index)\\n'\n"
           '        \'            elif hasattr(lyr, "attention") and hasattr(lyr.attention, "query") and hasattr(lyr, '
           '"ffn"):\\n\'\n'
           "        '                # ALBERT-style\\n'\n"
           "        '                if lyr.attention.query is not None:\\n'\n"
           "        '                    lyr.attention.query = prune_layer(lyr.attention.query, index, dim=1)\\n'\n"
           "        '                    lyr.attention.key   = prune_layer(lyr.attention.key,   index, dim=1)\\n'\n"
           "        '                if lyr.attention.value is not None:\\n'\n"
           "        '                    lyr.attention.value = prune_layer(lyr.attention.value, index, dim=1)\\n'\n"
           "        '                    lyr.attention.dense = prune_layer(lyr.attention.dense, index, dim=0)\\n'\n"
           "        '                    prune_layer_norm(lyr.attention.LayerNorm, index)\\n'\n"
           "        '                if lyr.ffn is not None:\\n'\n"
           "        '                    lyr.ffn        = prune_layer(lyr.ffn,        index, dim=1)\\n'\n"
           "        '                    lyr.ffn_output = prune_layer(lyr.ffn_output, index, dim=0)\\n'\n"
           "        '                    prune_layer_norm(lyr.full_layer_layer_norm, index)\\n'\n"
           '        \'            elif hasattr(lyr, "attention") and hasattr(lyr.attention, "q_lin"):\\n\'\n'
           "        '                # DistilBERT-style\\n'\n"
           "        '                if lyr.attention.q_lin is not None:\\n'\n"
           "        '                    lyr.attention.q_lin  = prune_layer(lyr.attention.q_lin,  index, dim=1)\\n'\n"
           "        '                    lyr.attention.k_lin  = prune_layer(lyr.attention.k_lin,  index, dim=1)\\n'\n"
           "        '                if lyr.attention.v_lin is not None:\\n'\n"
           "        '                    lyr.attention.v_lin   = prune_layer(lyr.attention.v_lin,   index, dim=1)\\n'\n"
           "        '                    lyr.attention.out_lin = prune_layer(lyr.attention.out_lin, index, dim=0)\\n'\n"
           "        '                    prune_layer_norm(lyr.attn_output.LayerNorm, index)\\n'\n"
           '        \'                if hasattr(lyr, "ffn_lin1") and lyr.ffn_lin1 is not None:\\n\'\n'
           "        '                    lyr.ffn_lin1 = prune_layer(lyr.ffn_lin1, index, dim=1)\\n'\n"
           "        '                    lyr.ffn_lin2 = prune_layer(lyr.ffn_lin2, index, dim=0)\\n'\n"
           "        '                    prune_layer_norm(lyr.ffn_output.LayerNorm, index)',\n"
           "        'cofi_utils: prune_model_with_z hidden_z loop model-agnostic')\n"
           '\n'
           '    patch_file(utils_path,\n'
           '        \'    if "hidden_z" in zs:\\n\'\n'
           '        \'        hidden_zs = zs["hidden_z"]\\n\'\n'
           "        '        index = torch.LongTensor(hidden_zs.squeeze().nonzero().squeeze().tolist())\\n'\n"
           "        '        index = index.to(model.device)\\n'\n"
           "        '\\n'\n"
           "        '        bert.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
           "        '            bert.embeddings.word_embeddings.weight.index_select(1, index).clone().detach())\\n'\n"
           "        '        bert.embeddings.word_embeddings.embedding_dim = index.shape[0]\\n'\n"
           "        '        bert.embeddings.position_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
           "        '            bert.embeddings.position_embeddings.weight.index_select(1, "
           "index).clone().detach())\\n'\n"
           "        '        bert.embeddings.position_embeddings.embedding_dim = index.shape[0]\\n'\n"
           "        '        bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
           "        '            bert.embeddings.token_type_embeddings.weight.index_select(1, "
           "index).clone().detach())\\n'\n"
           "        '        bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]\\n'\n"
           "        '        prune_layer_norm(bert.embeddings.LayerNorm, index)',\n"
           '        \'    if "hidden_z" in zs:\\n\'\n'
           '        \'        hidden_zs = zs["hidden_z"]\\n\'\n'
           "        '        index = torch.LongTensor(hidden_zs.squeeze().nonzero().squeeze().tolist())\\n'\n"
           "        '        index = index.to(model.device)\\n'\n"
           "        '\\n'\n"
           "        '        # ALBERT: embedding_size (128) != hidden_size (768) — skip embedding pruning.\\n'\n"
           "        '        # Prune embedding_hidden_mapping_in output dim instead (128->768 becomes 128->N).\\n'\n"
           '        \'        if hasattr(bert, "encoder") and hasattr(bert.encoder, '
           '"embedding_hidden_mapping_in"):\\n\'\n'
           "        '            bert.encoder.embedding_hidden_mapping_in = prune_linear_layer(\\n'\n"
           "        '                bert.encoder.embedding_hidden_mapping_in, index, dim=0)\\n'\n"
           "        '        _emb_dim = bert.embeddings.word_embeddings.weight.shape[1]\\n'\n"
           "        '        if _emb_dim == hidden_zs.squeeze().shape[0]:\\n'\n"
           "        '            bert.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
           "        '                bert.embeddings.word_embeddings.weight.index_select(1, "
           "index).clone().detach())\\n'\n"
           "        '            bert.embeddings.word_embeddings.embedding_dim = index.shape[0]\\n'\n"
           "        '            bert.embeddings.position_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
           "        '                bert.embeddings.position_embeddings.weight.index_select(1, "
           "index).clone().detach())\\n'\n"
           "        '            bert.embeddings.position_embeddings.embedding_dim = index.shape[0]\\n'\n"
           '        \'            if hasattr(bert.embeddings, "token_type_embeddings"):\\n\'\n'
           "        '                bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
           "        '                    bert.embeddings.token_type_embeddings.weight.index_select(1, "
           "index).clone().detach())\\n'\n"
           "        '                bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]\\n'\n"
           "        '            prune_layer_norm(bert.embeddings.LayerNorm, index)',\n"
           "        'cofi_utils: ALBERT skip embedding hidden_z + prune embedding_hidden_mapping_in')\n"
           '\n'
           '    patch_file(utils_path,\n'
           "        'def prune_intermediate_layers(model, keep_dims):\\n'\n"
           '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\\n\'\n'
           "        '    device = model.device\\n'\n"
           "        '    for layer in keep_dims:\\n'\n"
           "        '        if len(keep_dims[layer]) == 0:\\n'\n"
           "        '            bert.encoder.layer[layer].intermediate.dense = None\\n'\n"
           "        '            bert.encoder.layer[layer].output.dense = None\\n'\n"
           "        '        else:\\n'\n"
           "        '            bert.encoder.layer[layer].intermediate.dense = "
           'prune_linear_layer(bert.encoder.layer[layer].intermediate.dense, '
           "index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)\\n'\n"
           "        '            bert.encoder.layer[layer].output.dense = "
           'prune_linear_layer(bert.encoder.layer[layer].output.dense, '
           "index=torch.LongTensor(keep_dims[layer]).to(device), dim=1)',\n"
           "        'def prune_intermediate_layers(model, keep_dims):\\n'\n"
           '        \'    if hasattr(model, "bert"):\\n\'\n'
           "        '        bert = model.bert\\n'\n"
           '        \'    elif hasattr(model, "albert"):\\n\'\n'
           "        '        bert = model.albert\\n'\n"
           '        \'    elif hasattr(model, "roberta"):\\n\'\n'
           "        '        bert = model.roberta\\n'\n"
           '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
           "        '        bert = model.distilbert\\n'\n"
           "        '    else:\\n'\n"
           '        \'        raise ValueError(f"Unknown model type: {type(model)}")\\n\'\n'
           "        '    device = model.device\\n'\n"
           "        '    layers = _get_layers(bert)\\n'\n"
           "        '    for layer in keep_dims:\\n'\n"
           "        '        lyr = layers[min(layer, len(layers) - 1)]\\n'\n"
           '        \'        if hasattr(lyr, "intermediate"):\\n\'\n'
           "        '            if len(keep_dims[layer]) == 0:\\n'\n"
           "        '                lyr.intermediate.dense = None\\n'\n"
           "        '                lyr.output.dense = None\\n'\n"
           "        '            else:\\n'\n"
           "        '                idx = torch.LongTensor(keep_dims[layer]).to(device)\\n'\n"
           "        '                lyr.intermediate.dense = prune_linear_layer(lyr.intermediate.dense, index=idx, "
           "dim=0)\\n'\n"
           "        '                lyr.output.dense       = prune_linear_layer(lyr.output.dense,       index=idx, "
           "dim=1)\\n'\n"
           '        \'        elif hasattr(lyr, "ffn"):\\n\'\n'
           "        '            if len(keep_dims[layer]) == 0:\\n'\n"
           "        '                lyr.ffn = lyr.ffn_output = None\\n'\n"
           "        '            else:\\n'\n"
           "        '                idx = torch.LongTensor(keep_dims[layer]).to(device)\\n'\n"
           "        '                lyr.ffn        = prune_linear_layer(lyr.ffn,        index=idx, dim=0)\\n'\n"
           "        '                lyr.ffn_output = prune_linear_layer(lyr.ffn_output, index=idx, dim=1)\\n'\n"
           '        \'        elif hasattr(lyr, "ffn_lin1"):\\n\'\n'
           "        '            if len(keep_dims[layer]) == 0:\\n'\n"
           "        '                lyr.ffn_lin1 = lyr.ffn_lin2 = None\\n'\n"
           "        '            else:\\n'\n"
           "        '                idx = torch.LongTensor(keep_dims[layer]).to(device)\\n'\n"
           "        '                lyr.ffn_lin1 = prune_linear_layer(lyr.ffn_lin1, index=idx, dim=0)\\n'\n"
           "        '                lyr.ffn_lin2 = prune_linear_layer(lyr.ffn_lin2, index=idx, dim=1)',\n"
           "        'cofi_utils: prune_intermediate_layers model-agnostic')\n"
           '\n'
           '    patch_file(utils_path,\n'
           "        '    for layer in range(0, 12):\\n'\n"
           '        \'        print("Layer:", layer)\\n\'\n'
           "        '        if bert.encoder.layer[layer].attention.self.query is not None:\\n'\n"
           '        \'            print("query:", bert.encoder.layer[layer].attention.self.query.weight.shape)\\n\'\n'
           '        \'            print("key:", bert.encoder.layer[layer].attention.self.key.weight.shape)\\n\'\n'
           "        '        else:\\n'\n"
           '        \'            print("query:", None)\\n\'\n'
           '        \'            print("key:", None)\\n\'\n'
           "        '        if bert.encoder.layer[layer].attention.self.value is not None:\\n'\n"
           '        \'            print("value:", bert.encoder.layer[layer].attention.self.value.weight.shape)\\n\'\n'
           '        \'            print("output:", '
           "bert.encoder.layer[layer].attention.output.dense.weight.shape)\\n'\n"
           "        '        else:\\n'\n"
           '        \'            print("value:", None)\\n\'\n'
           '        \'            print("output:", None)\\n\'\n'
           "        '        if bert.encoder.layer[layer].intermediate.dense is not None:\\n'\n"
           '        \'            print("up:", bert.encoder.layer[layer].intermediate.dense.weight.shape)\\n\'\n'
           '        \'            print("down:", bert.encoder.layer[layer].output.dense.weight.shape)\\n\'\n'
           "        '        else:\\n'\n"
           '        \'            print("up", None)\\n\'\n'
           '        \'            print("down", None)\',\n'
           "        '    layers = _get_layers(bert)\\n'\n"
           "        '    for layer in range(len(layers)):\\n'\n"
           '        \'        print("Layer:", layer)\\n\'\n'
           "        '        lyr = layers[layer]\\n'\n"
           '        \'        if hasattr(lyr, "attention") and hasattr(lyr.attention, "self"):\\n\'\n'
           "        '            q = lyr.attention.self.query\\n'\n"
           "        '            v = lyr.attention.self.value\\n'\n"
           '        \'            up = lyr.intermediate.dense if hasattr(lyr, "intermediate") else None\\n\'\n'
           '        \'            down = lyr.output.dense if hasattr(lyr, "output") else None\\n\'\n'
           '        \'        elif hasattr(lyr, "attention") and hasattr(lyr.attention, "query"):\\n\'\n'
           "        '            q = lyr.attention.query\\n'\n"
           "        '            v = lyr.attention.value\\n'\n"
           '        \'            up = lyr.ffn if hasattr(lyr, "ffn") else None\\n\'\n'
           '        \'            down = lyr.ffn_output if hasattr(lyr, "ffn_output") else None\\n\'\n'
           "        '        else:\\n'\n"
           '        \'            q = getattr(lyr.attention, "q_lin", None)\\n\'\n'
           '        \'            v = getattr(lyr.attention, "v_lin", None)\\n\'\n'
           '        \'            up = getattr(lyr, "ffn_lin1", None)\\n\'\n'
           '        \'            down = getattr(lyr, "ffn_lin2", None)\\n\'\n'
           '        \'        print("query:", q.weight.shape if q is not None else None)\\n\'\n'
           '        \'        print("value:", v.weight.shape if v is not None else None)\\n\'\n'
           '        \'        print("up:", up.weight.shape if up is not None else None)\\n\'\n'
           '        \'        print("down:", down.weight.shape if down is not None else None)\',\n'
           "        'cofi_utils: print loop model-agnostic')\n"
           '\n'
           '    # ── models/l0_module.py ────────────────────────────────────────────────────\n'
           "    print('\\n  Patching models/l0_module.py...')\n"
           "    l0_path = os.path.join(AL_REPO_DIR, 'models', 'l0_module.py')\n"
           '    patch_file(l0_path,\n'
           "        '        self.hidden_size = config.hidden_size\\n'\n"
           "        '        self.intermediate_size = config.intermediate_size \\n'\n"
           "        '        self.num_attention_heads = config.num_attention_heads\\n'\n"
           "        '        self.mlp_num_per_layer = 1\\n'\n"
           "        '        self.dim_per_head = self.hidden_size // self.num_attention_heads \\n'\n"
           "        '        self.num_hidden_layers = config.num_hidden_layers\\n'\n"
           "        '        self.vocab_size = config.vocab_size',\n"
           '        \'        self.hidden_size = getattr(config, "hidden_size", getattr(config, "dim", None))\\n\'\n'
           '        \'        self.intermediate_size = getattr(config, "intermediate_size", getattr(config, '
           '"hidden_dim", None))\\n\'\n'
           '        \'        self.num_attention_heads = getattr(config, "num_attention_heads", getattr(config, '
           '"n_heads", None))\\n\'\n'
           "        '        self.mlp_num_per_layer = 1\\n'\n"
           "        '        self.dim_per_head = self.hidden_size // self.num_attention_heads\\n'\n"
           '        \'        self.num_hidden_layers = getattr(config, "num_hidden_layers", getattr(config, '
           '"n_layers", None))\\n\'\n'
           "        '        self.vocab_size = config.vocab_size',\n"
           "        'l0_module: config getattr fallbacks')\n"
           '\n'
           '    # ── trainer/trainer.py ─────────────────────────────────────────────────────\n'
           "    print('\\n  Patching trainer/trainer.py...')\n"
           "    trainer_path = os.path.join(AL_REPO_DIR, 'trainer', 'trainer.py')\n"
           '\n'
           '    patch_file(trainer_path,\n'
           "        '* (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)',\n"
           "        '* (torch.distributed.get_world_size() if (self.args.local_rank != -1 and "
           "torch.distributed.is_initialized()) else 1)',\n"
           "        'trainer: distributed world_size guard')\n"
           '\n'
           '    patch_file(trainer_path,\n'
           "        '                if self.start_prune:\\n'\n"
           "        '                    zs = self.l0_module.forward(training=True) #! get the zs\\n'\n"
           "        '                    self.fill_inputs_with_zs(zs, inputs) #! use the zs',\n"
           "        '                if self.start_prune and self.l0_module is not None:\\n'\n"
           "        '                    zs = self.l0_module.forward(training=True) #! get the zs\\n'\n"
           "        '                    self.fill_inputs_with_zs(zs, inputs) #! use the zs',\n"
           "        'trainer: guard l0_module None in train loop')\n"
           '\n'
           '    patch_file(trainer_path,\n'
           "        '        lagrangian_loss = None\\n'\n"
           "        '        if self.start_prune:\\n'\n"
           "        '            lagrangian_loss, _, _ = \\\\\\n'\n"
           "        '                self.l0_module.lagrangian_regularization(\\n'\n"
           "        '                    self.global_step - self.prepruning_finetune_steps)\\n'\n"
           "        '            loss += lagrangian_loss',\n"
           "        '        lagrangian_loss = None\\n'\n"
           "        '        if self.start_prune and self.l0_module is not None:\\n'\n"
           "        '            lagrangian_loss, _, _ = \\\\\\n'\n"
           "        '                self.l0_module.lagrangian_regularization(\\n'\n"
           "        '                    self.global_step - self.prepruning_finetune_steps)\\n'\n"
           "        '            loss += lagrangian_loss',\n"
           "        'trainer: guard lagrangian_regularization')\n"
           '\n'
           '    patch_file(trainer_path,\n'
           "        '        if self.start_prune:\\n'\n"
           "        '            self.l0_module.eval()\\n'\n"
           "        '            zs = self.l0_module.forward(training=False)',\n"
           "        '        if self.start_prune and self.l0_module is not None:\\n'\n"
           "        '            self.l0_module.eval()\\n'\n"
           "        '            zs = self.l0_module.forward(training=False)',\n"
           "        'trainer: guard l0_module.eval() in prediction_loop')\n"
           '\n'
           '    patch_file(trainer_path,\n'
           '        \'        torch.save(self.l0_module, os.path.join(output_dir, "l0_module.pt"))\\n\'\n'
           "        '\\n'\n"
           "        '        zs = self.l0_module.forward(training=False)\\n'\n"
           '        \'        torch.save(zs, os.path.join(output_dir, "zs.pt"))\',\n'
           "        '        if self.l0_module is not None:\\n'\n"
           '        \'            torch.save(self.l0_module, os.path.join(output_dir, "l0_module.pt"))\\n\'\n'
           "        '            zs = self.l0_module.forward(training=False)\\n'\n"
           '        \'            torch.save(zs, os.path.join(output_dir, "zs.pt"))\',\n'
           "        'trainer: guard l0_module save')\n"
           '\n'
           '    patch_file(trainer_path,\n'
           "        '                else:\\n'\n"
           "        '                    specified_teacher_layers = [2, 5, 8, 11]',\n"
           "        '                else:\\n'\n"
           "        '                    n_teacher_layers = len(teacher_outputs[2]) - 1\\n'\n"
           "        '                    if n_teacher_layers >= 12:\\n'\n"
           "        '                        specified_teacher_layers = [2, 5, 8, 11]\\n'\n"
           "        '                    else:\\n'\n"
           "        '                        step = max(1, n_teacher_layers // 4)\\n'\n"
           "        '                        specified_teacher_layers = [\\n'\n"
           "        '                            min(i * step, n_teacher_layers - 1) for i in range(1, 5)]\\n'\n"
           "        '                        specified_teacher_layers = sorted(set(specified_teacher_layers))\\n'\n"
           "        '                        while len(specified_teacher_layers) < 4:\\n'\n"
           "        '                            specified_teacher_layers.append(n_teacher_layers - 1)',\n"
           "        'trainer: dynamic teacher layer indices')\n"
           '\n'
           '    patch_file(trainer_path,\n'
           "        '                elif self.additional_args.layer_distill_version in (3, 4, 5, 6):\\n'\n"
           "        '                    last_aligned_layer = 12\\n'\n"
           "        '                    alignment = []\\n'\n"
           "        '                    for search_index in range(len(specified_teacher_layers)-1, -1, -1):\\n'\n"
           "        '                        indexes = layerwiseloss[search_index].sort()[1]\\n'\n"
           "        '                        if existing_layers is not None:\\n'\n"
           "        '                            align = indexes[(\\n'\n"
           "        '                                indexes < last_aligned_layer) & existing_layers]\\n'\n"
           "        '                        else:\\n'\n"
           "        '                            align = indexes[indexes < last_aligned_layer]\\n'\n"
           "        '                        if len(align) > 0:\\n'\n"
           "        '                            align = align[0]\\n'\n"
           "        '                        else:\\n'\n"
           "        '                            align = last_aligned_layer\\n'\n"
           "        '                        alignment.append(align)\\n'\n"
           "        '                        last_aligned_layer = align\\n'\n"
           "        '                    alignment.reverse()\\n'\n"
           "        '                    alignment = torch.tensor(alignment).to(device)',\n"
           "        '                elif self.additional_args.layer_distill_version in (3, 4, 5, 6):\\n'\n"
           "        '                    n_student_layers = layerwiseloss.shape[1]\\n'\n"
           "        '                    last_aligned_layer = n_student_layers\\n'\n"
           "        '                    alignment = []\\n'\n"
           "        '                    for search_index in range(len(specified_teacher_layers)-1, -1, -1):\\n'\n"
           "        '                        indexes = layerwiseloss[search_index].sort()[1]\\n'\n"
           "        '                        if existing_layers is not None:\\n'\n"
           "        '                            _el = existing_layers[:n_student_layers]\\n'\n"
           "        '                            align = indexes[(\\n'\n"
           "        '                                indexes < last_aligned_layer) & _el]\\n'\n"
           "        '                        else:\\n'\n"
           "        '                            align = indexes[indexes < last_aligned_layer]\\n'\n"
           "        '                        if len(align) > 0:\\n'\n"
           "        '                            align = align[0]\\n'\n"
           "        '                        else:\\n'\n"
           "        '                            align = torch.tensor(n_student_layers - 1).to(indexes.device)\\n'\n"
           "        '                        alignment.append(align)\\n'\n"
           "        '                        last_aligned_layer = align\\n'\n"
           "        '                    alignment.reverse()\\n'\n"
           "        '                    alignment = torch.stack([a if torch.is_tensor(a) else torch.tensor(a) for a "
           "in alignment]).to(device)',\n"
           "        'trainer: clamp student layer alignment to actual layer count')\n"
           '\n'
           '    patch_file(trainer_path,\n'
           "        '                teacher_outputs = self.teacher_model(**teacher_inputs)',\n"
           '        \'                teacher_inputs["output_hidden_states"] = True\\n\'\n'
           "        '                teacher_outputs = self.teacher_model(**teacher_inputs)',\n"
           "        'trainer: force output_hidden_states=True for teacher')\n"
           '\n'
           '    patch_file(trainer_path,\n'
           "        '            if logits is not None:\\n'\n"
           "        '                preds_host = logits if preds_host is None else nested_concat(\\n'\n"
           "        '                    preds_host, logits)',\n"
           "        '            if logits is not None:\\n'\n"
           "        '                logits = logits.cpu() if torch.is_tensor(logits) else logits\\n'\n"
           "        '            preds_host = logits if preds_host is None else nested_concat(\\n'\n"
           "        '                    preds_host, logits)',\n"
           "        'trainer: move logits to CPU before concat')\n"
           '\n'
           '    patch_file(trainer_path,\n'
           "        '            if labels is not None:\\n'\n"
           "        '                labels_host = labels if labels_host is None else nested_concat(\\n'\n"
           "        '                    labels_host, labels)',\n"
           "        '            if labels is not None:\\n'\n"
           "        '                labels = labels.cpu() if torch.is_tensor(labels) else labels\\n'\n"
           "        '                labels_host = labels if labels_host is None else nested_concat(\\n'\n"
           "        '                    labels_host, labels)',\n"
           "        'trainer: move labels to CPU before concat')\n"
           '\n'
           "    print('\\nBlock 0 complete.')\n"
           '\n'
           '# ── Block 1: Download pre-tuned models ────────────────────────────────────────\n'
           'def block1(tasks):\n'
           '    import subprocess, sys\n'
           "    subprocess.run([sys.executable, '-m', 'pip', 'install', 'sentencepiece', '-q'], check=True)\n"
           "    header('BLOCK 1 — Download Pre-tuned ALBERT-base-v2 Models')\n"
           '    from transformers import AutoModelForSequenceClassification, AutoTokenizer\n'
           '\n'
           '    for task in tasks:\n'
           '        hf_id = AL_PRETRAINED_FT.get(task)\n'
           '        if not hf_id:\n'
           "            print(f'[SKIP] {task}: no HF ID set in AL_PRETRAINED_FT')\n"
           '            continue\n'
           '        out = ft_dir(task)\n'
           '        os.makedirs(out, exist_ok=True)\n'
           '        if model_saved(out):\n'
           "            print(f'[SKIP] {task}: already at {out}')\n"
           '            continue\n'
           "        print(f'Downloading {hf_id} -> {out} ...')\n"
           '        model = AutoModelForSequenceClassification.from_pretrained(\n'
           '            hf_id, trust_remote_code=True)\n'
           '        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)\n'
           '        model.save_pretrained(out)\n'
           '        tok.save_pretrained(out)\n'
           '        del model, tok\n'
           "        print(f'  Saved to {out}')\n"
           '\n'
           "    print('\\nBlock 1 done.')\n"
           '\n'
           '\n'
           '# ── Block 2: CoFi pruning ──────────────────────────────────────────────────────\n'
           'def block2(tasks):\n'
           "    header('BLOCK 2 — CoFi Pruning (ALBERT)')\n"
           '\n'
           '    if AL_REPO_DIR not in sys.path:\n'
           '        sys.path.insert(0, AL_REPO_DIR)\n'
           '\n'
           "    env = {**os.environ, 'HF_DATASETS_TRUST_REMOTE_CODE': '1', 'PYTORCH_CUDA_ALLOC_CONF': "
           "'expandable_segments:True'}\n"
           '\n'
           '    for task in tasks:\n'
           '        ft  = ft_dir(task)\n'
           '        out = pr_dir(task)\n'
           '        cfg = task_cfg(task)\n'
           '        os.makedirs(out, exist_ok=True)\n'
           '\n'
           '        if not model_saved(ft):\n'
           "            print(f'[ERROR] {task}: fine-tuned model missing at {ft}. Run --block 1 first.')\n"
           '            continue\n'
           '\n'
           "        best = os.path.join(out, 'best')\n"
           '        if model_saved(best):\n'
           "            print(f'[SKIP] {task}: already pruned at {best}')\n"
           '            continue\n'
           '\n'
           "        print(f'\\nPruning albert/{task} -> {out}')\n"
           "        log_file = os.path.join(out, 'pruning_log.txt')\n"
           "        print(f'Log: {log_file}  (tail -f {log_file})')\n"
           '\n'
           '        cmd = [\n'
           '            sys.executable,\n'
           "            os.path.join(AL_REPO_DIR, 'run_glue_prune.py'),\n"
           "            '--model_name_or_path',          ft,\n"
           "            '--task_name',                   task,\n"
           "            '--do_train', '--do_eval',\n"
           "            '--max_seq_length',              '128',\n"
           "            '--per_device_train_batch_size', '32',\n"
           "            '--per_device_eval_batch_size',  '32',\n"
           "            '--learning_rate',               '2e-5',\n"
           "            '--reg_learning_rate',           '0.1',\n"
           "            '--num_train_epochs',            str(cfg['prune_epochs']),\n"
           "            '--output_dir',                  out,\n"
           "            '--save_steps',                  str(cfg['save_steps']),\n"
           "            '--save_total_limit',            '2',\n"
           "            '--eval_steps',                  str(cfg['eval_steps']),\n"
           "            '--eval_strategy',               'steps',\n"
           "            '--seed',                        str(SEED),\n"
           "            '--pruning_type',     'structured_heads+structured_mlp+hidden+layer',\n"
           "            '--target_sparsity',             str(SPARSITY),\n"
           "            '--sparsity_epsilon',            '0.01',\n"
           "            '--freeze_embeddings',\n"
           "            '--do_distill', '--do_layer_distill',\n"
           "            '--distillation_path',           ft,\n"
           "            '--distill_ce_loss_alpha',       '0.1',\n"
           "            '--distill_loss_alpha',          '0.9',\n"
           "            '--distill_temp',                '2',\n"
           "            '--layer_distill_version',       str(cfg['layer_distill_v']),\n"
           "            '--prepruning_finetune_epochs',  str(cfg['prepruning']),\n"
           "            '--lagrangian_warmup_epochs',    str(cfg['lag_warmup']),\n"
           "            '--scheduler_type',              'linear',\n"
           "            '--local_rank',                  '-1',\n"
           "            '--report_to',                   'none',\n"
           '        ]\n'
           '\n'
           "        with open(log_file, 'w') as log:\n"
           '            proc = subprocess.Popen(\n'
           '                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n'
           '                text=True, cwd=AL_REPO_DIR, env=env)\n'
           '            for line in proc.stdout:\n'
           '                sys.stdout.write(line)\n'
           '                sys.stdout.flush()\n'
           '                log.write(line)\n'
           '                log.flush()\n'
           '            proc.wait()\n'
           '\n'
           '        if model_saved(best):\n'
           "            print(f'\\n[DONE] {task}: best model at {best}')\n"
           '        else:\n'
           "            print(f'\\n[WARNING] {task}: no best/ checkpoint. Check {log_file}')\n"
           '\n'
           "    print('\\nBlock 2 done.')\n"
           '\n'
           '\n'
           '# ── Block 3: Evaluation ────────────────────────────────────────────────────────\n'
           'def block3(tasks):\n'
           "    header('BLOCK 3 — Evaluation (ALBERT)')\n"
           '\n'
           '    import torch\n'
           '    from torch.utils.data import DataLoader\n'
           '    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig\n'
           '    from datasets import load_dataset\n'
           '    import evaluate as hf_evaluate\n'
           '\n'
           '    if AL_REPO_DIR not in sys.path:\n'
           '        sys.path.insert(0, AL_REPO_DIR)\n'
           '\n'
           "    # Compute ALBERT_BASE_PARAMS via repo's own calculate_parameters()\n"
           '    global ALBERT_BASE_PARAMS\n'
           '    if ALBERT_BASE_PARAMS is None:\n'
           '        try:\n'
           '            from utils.utils import calculate_parameters\n'
           '            cfg_tmp = AutoConfig.from_pretrained(\n'
           "                'albert-base-v2', num_labels=2, trust_remote_code=True)\n"
           '            m_tmp = AutoModelForSequenceClassification.from_config(cfg_tmp)\n'
           '            ALBERT_BASE_PARAMS = calculate_parameters(m_tmp)\n'
           "            print(f'[INFO] ALBERT_BASE_PARAMS (non-embedding) = {ALBERT_BASE_PARAMS:,}')\n"
           '            del m_tmp\n'
           '        except Exception as e:\n'
           "            print(f'[WARN] Could not auto-compute ALBERT_BASE_PARAMS: {e}')\n"
           '            ALBERT_BASE_PARAMS = 1  # avoid div-by-zero; sparsity will show 0%\n'
           '\n'
           '    TASK_KEYS = {\n'
           "        'sst2': ('sentence',  None),\n"
           "        'qnli': ('question',  'sentence'),\n"
           "        'mnli': ('premise',   'hypothesis'),\n"
           "        'qqp':  ('question1', 'question2'),\n"
           "        'rte':  ('sentence1', 'sentence2'),\n"
           '    }\n'
           '\n'
           '    def run_evaluation(model_path, task, label, out_dir):\n'
           '        result_file = eval_path(out_dir)\n'
           '        if os.path.exists(result_file):\n'
           "            print(f'[SKIP] Already evaluated: {result_file}')\n"
           '            r = json.load(open(result_file))\n'
           "            for k, v in r.items(): print(f'  {k}: {v}')\n"
           '            return r\n'
           '\n'
           "        print(f'\\nEvaluating [{label}] on {task.upper()} ...')\n"
           "        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
           '\n'
           "        tok_path = model_path if os.path.exists(os.path.join(model_path, 'tokenizer_config.json')) else "
           'ft_dir(task)\n'
           '        tok = AutoTokenizer.from_pretrained(tok_path)\n'
           '\n'
           '        # [HIGH] Use exact prefix match for pruned detection — NOT substring —\n'
           '        # because "unpruned" contains "pruned" as a substring (context doc s2b).\n'
           "        if label.lower().strip().startswith('albert pruned'):\n"
           '            from models.modeling_albert import CoFiAlbertForSequenceClassification\n'
           '            from utils.cofi_utils import load_zs, load_model\n'
           '            from utils.utils import calculate_parameters\n'
           '            zs = load_zs(model_path)\n'
           '            if zs is not None:\n'
           '                model = load_model(model_path, CoFiAlbertForSequenceClassification, zs)\n'
           '            else:\n'
           '                model = CoFiAlbertForSequenceClassification.from_pretrained(model_path)\n'
           '        else:\n'
           '            model = AutoModelForSequenceClassification.from_pretrained(model_path)\n'
           '\n'
           '        n_params = sum(p.numel() for p in model.parameters())\n'
           '        mem_mb   = n_params * 4 / 1e6\n'
           '        model    = model.to(device).eval()\n'
           '\n'
           "        split = 'validation_matched' if task == 'mnli' else 'validation'\n"
           "        ds    = load_dataset('glue', task, trust_remote_code=True)[split]\n"
           '        col_a, col_b = TASK_KEYS[task]\n'
           '        ds = ds.select(range(min(1000, len(ds))))\n'
           '\n'
           '        def tokenize(batch):\n'
           '            args = ((batch[col_a],) if col_b is None\n'
           '                    else (batch[col_a], batch[col_b]))\n'
           "            return tok(*args, padding='max_length', truncation=True, max_length=128)\n"
           '\n'
           '        ds = ds.map(tokenize, batched=True,\n'
           '                    remove_columns=[c for c in ds.column_names\n'
           "                                    if c not in ['label', 'labels', 'idx']])\n"
           "        ds.set_format('torch')\n"
           '        loader = DataLoader(ds, batch_size=32)\n'
           '\n'
           '        all_preds, all_labels = [], []\n'
           '        total_time, total_examples = 0.0, 0\n'
           '\n'
           '        # Warmup pass\n'
           '        with torch.no_grad():\n'
           '            for batch in loader:\n'
           '                inp = {k: v.to(device) for k, v in batch.items()\n'
           "                       if k in ['input_ids', 'attention_mask', 'token_type_ids']}\n"
           '                model(**inp); break\n'
           '\n'
           '        with torch.no_grad():\n'
           '            for batch in loader:\n'
           "                labels = batch.pop('label', batch.pop('labels', None))\n"
           '                inp = {k: v.to(device) for k, v in batch.items()\n'
           "                       if k in ['input_ids', 'attention_mask', 'token_type_ids']}\n"
           '                if torch.cuda.is_available(): torch.cuda.synchronize()\n'
           '                t0  = time.perf_counter()\n'
           '                out = model(**inp)\n'
           "                logits = out.logits if hasattr(out, 'logits') else out[0]\n"
           '                if torch.cuda.is_available(): torch.cuda.synchronize()\n'
           '                total_time     += time.perf_counter() - t0\n'
           '                total_examples += logits.shape[0]\n'
           '                all_preds.extend(logits.argmax(-1).cpu().numpy().tolist())\n'
           '                if labels is not None:\n'
           '                    all_labels.extend(labels.cpu().numpy().tolist())\n'
           '\n'
           '        latency_ms = (total_time / total_examples) * 1000\n'
           '        throughput  = total_examples / total_time\n'
           '\n'
           "        if task == 'qqp':\n"
           "            score = hf_evaluate.load('glue', 'qqp').compute(\n"
           "                predictions=all_preds, references=all_labels)['f1']\n"
           "            metric_name = 'F1'\n"
           "        elif task == 'mnli':\n"
           "            score = hf_evaluate.load('glue', 'mnli').compute(\n"
           "                predictions=all_preds, references=all_labels)['accuracy']\n"
           "            metric_name = 'Accuracy'\n"
           '        else:\n'
           "            score = hf_evaluate.load('glue', task).compute(\n"
           "                predictions=all_preds, references=all_labels)['accuracy']\n"
           "            metric_name = 'Accuracy'\n"
           '\n'
           "        # Sparsity via repo's calculate_parameters (excludes embeddings/classifier/pooler)\n"
           '        try:\n'
           '            from utils.utils import calculate_parameters\n'
           '            n_non_emb    = calculate_parameters(model)\n'
           '            sparsity_pct = max(0.0, (1 - n_non_emb / ALBERT_BASE_PARAMS) * 100)\n'
           '        except Exception:\n'
           '            sparsity_pct = 0.0\n'
           '\n'
           '        results = {\n'
           "            'label':          label,\n"
           "            'task':           task,\n"
           "            'n_params':       n_params,\n"
           "            'memory_mb':      round(mem_mb, 2),\n"
           "            'latency_ms':     round(latency_ms, 4),\n"
           "            'throughput_eps': round(throughput, 2),\n"
           "            'sparsity_pct':   round(sparsity_pct, 2),\n"
           '            metric_name:      round(score, 4),\n'
           '        }\n'
           '\n'
           '        W = 52\n'
           "        print('=' * W)\n"
           "        print(f'  {label} — {task.upper()}')\n"
           "        print('=' * W)\n"
           "        print(f'  {metric_name:<24}: {score:.4f}')\n"
           "        print(f'  Sparsity %             : {sparsity_pct:.1f}%')\n"
           "        print(f'  Parameters             : {n_params:,}')\n"
           "        print(f'  Memory MB              : {mem_mb:.1f}')\n"
           "        print(f'  Latency ms/example     : {latency_ms:.3f}')\n"
           "        print(f'  Throughput ex/sec      : {throughput:.1f}')\n"
           "        print('=' * W)\n"
           '\n'
           '        os.makedirs(out_dir, exist_ok=True)\n'
           "        with open(result_file, 'w') as f:\n"
           '            json.dump(results, f, indent=2)\n'
           "        print(f'  Saved: {result_file}')\n"
           '\n'
           '        del model\n'
           '        if torch.cuda.is_available(): torch.cuda.empty_cache()\n'
           '        return results\n'
           '\n'
           '    for task in tasks:\n'
           "        print(f'\\n--- {task.upper()} unpruned ---')\n"
           "        run_evaluation(ft_dir(task), task, 'ALBERT unpruned', ft_dir(task))\n"
           '\n'
           "        print(f'\\n--- {task.upper()} pruned 60% ---')\n"
           "        best = os.path.join(pr_dir(task), 'best')\n"
           '        if not model_saved(best):\n'
           "            print(f'  Not found: {best}. Run --block 2 --task {task} first.')\n"
           '        else:\n'
           "            run_evaluation(best, task, 'ALBERT pruned 60%', pr_dir(task))\n"
           '\n'
           "    print('\\nBlock 3 done.')\n"
           '\n'
           '\n'
           '# ── Block 4: Results table ─────────────────────────────────────────────────────\n'
           'def block4():\n'
           "    header('BLOCK 4 — Full Results Table (ALBERT)')\n"
           '\n'
           '    metric_label = {\n'
           "        'sst2': 'Accuracy', 'qnli': 'Accuracy', 'mnli': 'Accuracy',\n"
           "        'qqp': 'F1', 'rte': 'Accuracy',\n"
           '    }\n'
           '\n'
           '    def load_result(path):\n'
           '        p = eval_path(path)\n'
           '        if not os.path.exists(p): return None\n'
           '        return json.load(open(p))\n'
           '\n'
           '    W = 97\n'
           "    print('=' * W)\n"
           "    print('  COFI PRUNING RESULTS — ALBERT-base-v2 on GLUE (60% Sparsity)')\n"
           "    print('=' * W)\n"
           '    print(f"  {\'Task\':<6} {\'Model\':<26} {\'Score\':>8} {\'Mem MB\':>9} "\n'
           '          f"{\'Lat ms\':>9} {\'Tput ex/s\':>11} {\'Sparsity\':>10}")\n'
           "    print('-' * W)\n"
           '\n'
           '    for task in ALL_TASKS:\n'
           '        ml  = metric_label[task]\n'
           '        unp = load_result(ft_dir(task))\n'
           '        pru = load_result(pr_dir(task))\n'
           '\n'
           '        def fmt(r):\n'
           "            if r is None: return ['N/A'] * 5\n"
           "            score = r.get(ml, 'N/A')\n"
           '            return [\n'
           "                f'{score:.4f}' if isinstance(score, float) else str(score),\n"
           "                str(r.get('memory_mb', 'N/A')),\n"
           "                str(r.get('latency_ms', 'N/A')),\n"
           "                str(r.get('throughput_eps', 'N/A')),\n"
           '                f"{r.get(\'sparsity_pct\', \'N/A\')}%",\n'
           '            ]\n'
           '\n'
           '        u, p = fmt(unp), fmt(pru)\n'
           '        print(f"  {task:<6} {\'ALBERT unpruned\':<26} {u[0]:>8} {u[1]:>9} "\n'
           '              f"{u[2]:>9} {u[3]:>11} {u[4]:>10}")\n'
           '        print(f"  {\'\':<6} {\'ALBERT pruned 60%\':<26} {p[0]:>8} {p[1]:>9} "\n'
           '              f"{p[2]:>9} {p[3]:>11} {p[4]:>10}")\n'
           '        if unp and pru:\n'
           '            try:\n'
           "                speedup   = unp['latency_ms'] / pru['latency_ms']\n"
           '                retention = float(p[0]) / float(u[0]) * 100\n'
           '                print(f"  {\'\':<6}   speedup {speedup:.2f}x | score retention {retention:.1f}%")\n'
           '            except Exception:\n'
           '                pass\n'
           "        print('-' * W)\n"
           '\n'
           "    print('=' * W)\n"
           "    print('  Accuracy for SST-2/QNLI/MNLI/RTE; F1 for QQP')\n"
           "    print('  Sparsity: non-embedding params vs ALBERT-base-v2 baseline')\n"
           '\n'
           '\n'
           '# ── Status ─────────────────────────────────────────────────────────────────────\n'
           'def show_status():\n'
           "    header('STATUS — ALBERT')\n"
           '    W = 62\n'
           '    print(f"  {\'\':18}" + \'  \'.join(f\'{t:<6}\' for t in ALL_TASKS))\n'
           "    print('-' * W)\n"
           '    for label, fn in [\n'
           "        ('FT downloaded', ft_dir),\n"
           "        ('Pruned',        lambda t: os.path.join(pr_dir(t), 'best')),\n"
           "        ('Eval unpruned', lambda t: ft_dir(t)),\n"
           "        ('Eval pruned',   pr_dir),\n"
           '    ]:\n'
           "        if 'Eval' in label:\n"
           "            row = [('v' if os.path.exists(eval_path(fn(t))) else '.') for t in ALL_TASKS]\n"
           '        else:\n'
           "            row = [('v' if model_saved(fn(t)) else '.') for t in ALL_TASKS]\n"
           '        print(f"  {label:<18}" + \'  \'.join(f\'{s:<6}\' for s in row))\n'
           "    print('=' * W)\n"
           "    print('  v = done   . = not yet')\n"
           '\n'
           '\n'
           '# ── Main ───────────────────────────────────────────────────────────────────────\n'
           "if __name__ == '__main__':\n"
           "    parser = argparse.ArgumentParser(description='CoFiPruning ALBERT-base-v2 on GLUE')\n"
           "    parser.add_argument('--block',  type=int, choices=[0, 1, 2, 3, 4])\n"
           "    parser.add_argument('--task',   type=str, choices=ALL_TASKS)\n"
           "    parser.add_argument('--status', action='store_true')\n"
           '    args = parser.parse_args()\n'
           '\n'
           '    tasks = [args.task] if args.task else ALL_TASKS\n'
           '\n'
           '    if args.status:      show_status()\n'
           '    elif args.block == 0: block0()\n'
           '    elif args.block == 1: block1(tasks)\n'
           '    elif args.block == 2: block2(tasks)\n'
           '    elif args.block == 3: block3(tasks)\n'
           '    elif args.block == 4: block4()\n'
           '    else:                 parser.print_help()',
 'bertbase': '"""\n'
             'CoFiPruning — BERT-base on 5 GLUE Tasks\n'
             '=========================================\n'
             'Server path: /media/shared/Devshree/edgellm3/cofi_experiments\n'
             'CoFiPruning: /media/shared/Devshree/edgellm3/cofi_experiments/CoFiPruning\n'
             '\n'
             'Usage:\n'
             '  python cofi_base.py --status                  # show what is done\n'
             '  python cofi_base.py --block 0                 # patch repo (run once, safe to re-run)\n'
             '  python cofi_base.py --block 1                 # download all 5 pre-tuned models\n'
             '  python cofi_base.py --block 1 --task rte      # download one task only\n'
             '  python cofi_base.py --block 2                 # prune all 5 tasks sequentially\n'
             '  python cofi_base.py --block 2 --task rte      # prune one task only\n'
             '  python cofi_base.py --block 3                 # evaluate all (pruned + unpruned)\n'
             '  python cofi_base.py --block 3 --task rte      # evaluate one task only\n'
             '  python cofi_base.py --block 4                 # print full results table\n'
             '"""\n'
             '\n'
             'import argparse\n'
             'import json\n'
             'import os\n'
             'import re\n'
             'import subprocess\n'
             'import sys\n'
             'import time\n'
             '\n'
             '# ── Paths ──────────────────────────────────────────────────────────────────────\n'
             'BASE_DIR = os.path.dirname(os.path.abspath(__file__))\n'
             "BASE_REPO_DIR = os.path.join(BASE_DIR, 'CoFiPruning_Base')\n"
             '\n'
             '# ── Constants ──────────────────────────────────────────────────────────────────\n'
             'SPARSITY    = 0.6\n'
             'SEED        = 57\n'
             "ALL_TASKS   = ['sst2', 'qnli', 'mnli', 'qqp', 'rte']\n"
             "SMALL_TASKS = {'rte', 'mrpc', 'cola', 'stsb'}\n"
             '\n'
             'BASE_PRETRAINED_FT = {\n'
             "    'sst2': 'JeremiahZ/bert-base-uncased-sst2',\n"
             "    'qnli': 'JeremiahZ/bert-base-uncased-qnli',\n"
             "    'mnli': 'JeremiahZ/bert-base-uncased-mnli',\n"
             "    'qqp' : 'JeremiahZ/bert-base-uncased-qqp',\n"
             "    'rte' : 'JeremiahZ/bert-base-uncased-rte',\n"
             '}\n'
             '\n'
             'BERT_BASE_PARAMS = 85_054_464\n'
             '\n'
             '# ── Helpers ────────────────────────────────────────────────────────────────────\n'
             'def ft_dir(task):\n'
             "    return os.path.join(BASE_DIR, f'ft_base_{task}')\n"
             '\n'
             'def pr_dir(task):\n'
             "    return os.path.join(BASE_DIR, f'pr_base_{task}_s{int(SPARSITY*100)}')\n"
             '\n'
             'def eval_path(d):\n'
             "    return os.path.join(d, 'eval_results.json')\n"
             '\n'
             'def model_saved(path):\n'
             "    return (os.path.exists(os.path.join(path, 'pytorch_model.bin')) or\n"
             "            os.path.exists(os.path.join(path, 'model.safetensors')))\n"
             '\n'
             'def task_cfg(task):\n'
             '    if task in SMALL_TASKS:\n'
             '        return dict(prune_epochs=100, eval_steps=50, save_steps=50,\n'
             '                    prepruning=4, lag_warmup=20, layer_distill_v=4)\n'
             '    return dict(prune_epochs=20, eval_steps=500, save_steps=500,\n'
             '                prepruning=1, lag_warmup=2, layer_distill_v=3)\n'
             '\n'
             'def header(msg):\n'
             '    W = 60\n'
             "    print('\\n' + '=' * W)\n"
             "    print(f'  {msg}')\n"
             "    print('=' * W)\n"
             '\n'
             'def patch_file(fpath, old, new, description):\n'
             '    """\n'
             '    Idempotent patch.\n'
             '    - If new is already present: skip (already patched correctly)\n'
             '    - If old is not present: skip (not found)\n'
             '    - If old is present and new is absent: apply\n'
             '    Never uses try/except blocks in replacements — always direct substitution.\n'
             '    """\n'
             '    txt = open(fpath).read()\n'
             '    if new and new in txt:\n'
             "        print(f'  [already patched] {description}')\n"
             '        return\n'
             '    if old not in txt:\n'
             "        print(f'  [not found]       {description}')\n"
             '        return\n'
             "    open(fpath, 'w').write(txt.replace(old, new))\n"
             "    print(f'  [patched]         {description}')\n"
             '\n'
             '# ── Block 0: Patch repo ────────────────────────────────────────────────────────\n'
             'def block0():\n'
             '\n'
             "    header('BLOCK 0 — Fresh clone + patch CoFiPruning for BertBASE')\n"
             '\n'
             '    # ── Fresh clone ────────────────────────────────────────────────────────────\n'
             '    if os.path.exists(BASE_REPO_DIR):\n'
             "        print(f'Removing existing {BASE_REPO_DIR} ...')\n"
             "        subprocess.run(['rm', '-rf', BASE_REPO_DIR], check=True)\n"
             "    print('Cloning CoFiPruning ...')\n"
             '    subprocess.run(\n'
             "        ['git', 'clone', 'https://github.com/princeton-nlp/CoFiPruning.git', BASE_REPO_DIR],\n"
             '        check=True)\n'
             "    print('Clone done.\\n')\n"
             '\n'
             '    # ── Walk all .py files ─────────────────────────────────────────────────────\n'
             '    # Each patch_file call is a direct string substitution — no try/except blocks.\n'
             '    # All replacements are simple one-liners that cannot produce syntax errors.\n'
             '\n'
             "    print('\\n  Scanning all .py files...')\n"
             '    for root, dirs, files in os.walk(BASE_REPO_DIR):\n'
             "        dirs[:] = [d for d in dirs if d != '.git']\n"
             '        for fname in files:\n'
             "            if not fname.endswith('.py'):\n"
             '                continue\n'
             '            fpath = os.path.join(root, fname)\n'
             '            rel   = os.path.relpath(fpath, BASE_REPO_DIR)\n'
             '\n'
             '            # cached_path: both variants map to the same clean one-liner\n'
             '            patch_file(\n'
             '                fpath,\n'
             "                old='from transformers.file_utils import hf_bucket_url, cached_path',\n"
             "                new='from huggingface_hub import cached_download as cached_path',\n"
             "                description=f'{rel}: hf_bucket_url+cached_path -> huggingface_hub',\n"
             '            )\n'
             '            patch_file(\n'
             '                fpath,\n'
             "                old='from transformers.file_utils import cached_path',\n"
             "                new='from huggingface_hub import cached_download as cached_path',\n"
             "                description=f'{rel}: cached_path -> huggingface_hub',\n"
             '            )\n'
             '            patch_file(\n'
             '                fpath,\n'
             "                old='from transformers.file_utils import hf_bucket_url',\n"
             "                new='',\n"
             "                description=f'{rel}: remove hf_bucket_url',\n"
             '            )\n'
             '            patch_file(\n'
             '                fpath,\n'
             "                old='from datasets import load_dataset, load_metric, DatasetDict',\n"
             "                new='from datasets import load_dataset, DatasetDict\\nimport evaluate',\n"
             "                description=f'{rel}: load_metric -> evaluate',\n"
             '            )\n'
             '            patch_file(\n'
             '                fpath,\n'
             '                old=\'metric = load_metric("glue", data_args.task_name)\',\n'
             '                new=\'metric = evaluate.load("glue", data_args.task_name)\',\n'
             '                description=f\'{rel}: load_metric("glue") -> evaluate.load\',\n'
             '            )\n'
             '            patch_file(\n'
             '                fpath,\n'
             '                old=\'metric = load_metric("accuracy")\',\n'
             '                new=\'metric = evaluate.load("accuracy")\',\n'
             '                description=f\'{rel}: load_metric("accuracy") -> evaluate.load\',\n'
             '            )\n'
             '            patch_file(\n'
             '                fpath,\n'
             "                old='from black import main',\n"
             "                new='',\n"
             "                description=f'{rel}: remove black import',\n"
             '            )\n'
             '\n'
             '    # ── run_glue_prune.py ──────────────────────────────────────────────────────\n'
             "    glue_path = os.path.join(BASE_REPO_DIR, 'run_glue_prune.py')\n"
             "    print('\\n  Patching run_glue_prune.py...')\n"
             '    patch_file(\n'
             '        glue_path,\n'
             '        old=\'load_dataset("glue", data_args.task_name)\',\n'
             '        new=\'load_dataset("glue", data_args.task_name, trust_remote_code=True)\',\n'
             "        description='add trust_remote_code',\n"
             '    )\n'
             '    patch_file(\n'
             '        glue_path,\n'
             '        old=\'"evaluation_strategy"\',\n'
             '        new=\'"eval_strategy"\',\n'
             "        description='evaluation_strategy -> eval_strategy',\n"
             '    )\n'
             '\n'
             '    # ── modeling_bert.py: remove broken from_pretrained override ──────────────\n'
             "    print('\\n  Patching models/modeling_bert.py...')\n"
             "    bert_path = os.path.join(BASE_REPO_DIR, 'models/modeling_bert.py')\n"
             '    src = open(bert_path).read()\n'
             '    pat = re.compile(\n'
             "        r'[ \\t]*@classmethod\\s*\\n[ \\t]*def from_pretrained\\(cls.*?(?=\\n[ \\t]{0,4}(?:def |class "
             "|\\Z))',\n"
             '        re.DOTALL)\n'
             '    if re.search(pat, src):\n'
             "        open(bert_path, 'w').write(re.sub(pat, '', src))\n"
             "        print('  [patched]         modeling_bert.py: removed from_pretrained override')\n"
             '    else:\n'
             "        print('  [already patched] modeling_bert.py: no from_pretrained override found')\n"
             '\n'
             '    # ── trainer/trainer.py ─────────────────────────────────────────────────────\n'
             "    print('\\n  Patching trainer/trainer.py...')\n"
             "    trainer_path = os.path.join(BASE_REPO_DIR, 'trainer/trainer.py')\n"
             '    patch_file(\n'
             '        trainer_path,\n'
             "        old='* (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)',\n"
             "        new='',\n"
             "        description='remove world_size multiplication',\n"
             '    )\n'
             '    patch_file(\n'
             '        trainer_path,\n'
             "        old='if self.start_prune:\\n                zs = self.l0_module',\n"
             "        new='if self.start_prune and self.l0_module is not None:\\n                zs = "
             "self.l0_module',\n"
             "        description='guard l0_module None check',\n"
             '    )\n'
             '    patch_file(\n'
             '        trainer_path,\n'
             '        old="torch.save(self.l0_module, os.path.join(output_dir, \'l0_module.pt\'))",\n'
             '        new="if self.l0_module is not None:\\n            torch.save(self.l0_module, '
             'os.path.join(output_dir, \'l0_module.pt\'))",\n'
             "        description='guard l0_module save',\n"
             '    )\n'
             '\n'
             '    #utils/cofi_utils.py: remove broken load_model override\n'
             "    print('\\n  Patching utils/cofi_utils.py...')\n"
             "    utils_path = os.path.join(BASE_REPO_DIR, 'utils/cofi_utils.py')\n"
             '    patch_file(\n'
             '        utils_path,\n'
             '    old = \'    p = os.path.join(model_path, "pytorch_model.bin")\\n    loaded_weights = torch.load(p, '
             'map_location="cpu")\',\n'
             '    new =\n'
             '    \'    p_bin = os.path.join(model_path, "pytorch_model.bin")\\n\'\n'
             '    \'    p_safe = os.path.join(model_path, "model.safetensors")\\n\'\n'
             "    '    if os.path.exists(p_bin):\\n'\n"
             '    \'        loaded_weights = torch.load(p_bin, map_location="cpu")\\n\'\n'
             "    '    elif os.path.exists(p_safe):\\n'\n"
             "    '        from safetensors.torch import load_file\\n'\n"
             "    '        loaded_weights = load_file(p_safe)\\n'\n"
             "    '    else:\\n'\n"
             '    \'        raise FileNotFoundError(f"No model weights found in {model_path}")\',\n'
             "        description='utils/cofi_utils.py: load_model supports both .bin and .safetensors',\n"
             '    )\n'
             '\n'
             "    print('\\nAll patches done.')\n"
             "    print('Verify with: python cofi_base.py --block 0  (re-running should show all [already patched])')\n"
             '\n'
             '# ── Block 1: Download pre-tuned models ────────────────────────────────────────\n'
             'def block1(tasks):\n'
             "    header('BLOCK 1 — Download Pre-tuned Models')\n"
             '    from transformers import AutoModelForSequenceClassification, AutoTokenizer\n'
             '\n'
             '    for task in tasks:\n'
             '        out = ft_dir(task)\n'
             '        os.makedirs(out, exist_ok=True)\n'
             '        if model_saved(out):\n'
             "            print(f'[SKIP] {task}: already at {out}')\n"
             '            continue\n'
             '        hf_id = BASE_PRETRAINED_FT[task]\n'
             "        print(f'Downloading {hf_id} -> {out} ...')\n"
             '        model = AutoModelForSequenceClassification.from_pretrained(\n'
             '            hf_id, trust_remote_code=True)\n'
             '        tok = AutoTokenizer.from_pretrained(\n'
             '            hf_id, trust_remote_code=True)\n'
             '        model.save_pretrained(out)\n'
             '        tok.save_pretrained(out)\n'
             '        del model, tok\n'
             "        print(f'  Saved.')\n"
             '\n'
             "    print('\\nBlock 1 done.')\n"
             '\n'
             '# ── Block 2: CoFi pruning ──────────────────────────────────────────────────────\n'
             'def block2(tasks):\n'
             "    header('BLOCK 2 — CoFi Pruning')\n"
             '\n'
             '    if BASE_REPO_DIR not in sys.path:\n'
             '        sys.path.insert(0, BASE_REPO_DIR)\n'
             '\n'
             "    env = {**os.environ, 'HF_DATASETS_TRUST_REMOTE_CODE': '1'}\n"
             '\n'
             '    for task in tasks:\n'
             '        ft  = ft_dir(task)\n'
             '        out = pr_dir(task)\n'
             '        cfg = task_cfg(task)\n'
             '        os.makedirs(out, exist_ok=True)\n'
             '\n'
             '        if not model_saved(ft):\n'
             "            print(f'[ERROR] {task}: fine-tuned model missing at {ft}. Run --block 1 first.')\n"
             '            continue\n'
             '\n'
             "        best = os.path.join(out, 'best')\n"
             '        if model_saved(best):\n'
             "            print(f'[SKIP] {task}: already pruned at {best}')\n"
             '            continue\n'
             '\n'
             "        print(f'\\nPruning base/{task} -> {out}')\n"
             "        log_file = os.path.join(out, 'pruning_log.txt')\n"
             "        print(f'Log:     {log_file}')\n"
             "        print(f'Monitor: tail -f {log_file}')\n"
             "        print('Ctrl+C stops safely — resumes from last checkpoint on next run.\\n')\n"
             '\n'
             '        cmd = [\n'
             '            sys.executable,\n'
             "            os.path.join(BASE_REPO_DIR, 'run_glue_prune.py'),\n"
             "            '--model_name_or_path', 'bert-base-uncased',\n"
             "            '--task_name', task,\n"
             "            '--do_train', '--do_eval',\n"
             "            '--max_seq_length', '128',\n"
             "            '--per_device_train_batch_size', '32',\n"
             "            '--per_device_eval_batch_size', '32',\n"
             "            '--learning_rate', '2e-5',\n"
             "            '--reg_learning_rate', '0.01',\n"
             "            '--num_train_epochs', str(cfg['prune_epochs']),\n"
             "            '--output_dir', out,\n"
             "            '--save_steps', str(cfg['save_steps']),\n"
             "            '--save_total_limit', '2',\n"
             "            '--eval_steps', str(cfg['eval_steps']),\n"
             "            '--eval_strategy', 'steps',\n"
             "            '--seed', str(SEED),\n"
             "            '--pruning_type', 'structured_heads+structured_mlp+hidden+layer',\n"
             "            '--target_sparsity', str(SPARSITY),\n"
             "            '--sparsity_epsilon', '0.01',\n"
             "            '--freeze_embeddings',\n"
             "            '--do_distill', '--do_layer_distill',\n"
             "            '--distillation_path', ft,\n"
             "            '--distill_ce_loss_alpha', '0.1',\n"
             "            '--distill_loss_alpha', '0.9',\n"
             "            '--distill_temp', '2',\n"
             "            '--layer_distill_version', str(cfg['layer_distill_v']),\n"
             "            '--prepruning_finetune_epochs', str(cfg['prepruning']),\n"
             "            '--lagrangian_warmup_epochs', str(cfg['lag_warmup']),\n"
             "            '--scheduler_type', 'linear',\n"
             "            '--local_rank', '-1',\n"
             "            '--report_to', 'none',\n"
             '        ]\n'
             '\n'
             "        with open(log_file, 'w') as log:\n"
             '            proc = subprocess.Popen(\n'
             '                cmd,\n'
             '                stdout=subprocess.PIPE,\n'
             '                stderr=subprocess.STDOUT,\n'
             '                text=True,\n'
             '                cwd=BASE_REPO_DIR,\n'
             '                env=env,\n'
             '            )\n'
             '            for line in proc.stdout:\n'
             '                sys.stdout.write(line)\n'
             '                sys.stdout.flush()\n'
             '                log.write(line)\n'
             '                log.flush()\n'
             '            proc.wait()\n'
             '\n'
             '        if model_saved(best):\n'
             "            print(f'\\n[DONE] {task}: best model at {best}')\n"
             '        else:\n'
             "            print(f'\\n[WARNING] {task}: no best/ checkpoint. Check {log_file}')\n"
             '\n'
             "    print('\\nBlock 2 done.')\n"
             '\n'
             '# ── Block 3: Evaluation ────────────────────────────────────────────────────────\n'
             'def block3(tasks):\n'
             "    header('BLOCK 3 — Evaluation')\n"
             '\n'
             '    import torch\n'
             '    from torch.utils.data import DataLoader\n'
             '    from transformers import AutoModelForSequenceClassification, AutoTokenizer\n'
             '    from datasets import load_dataset\n'
             '    import evaluate as hf_evaluate\n'
             '\n'
             '    def run_evaluation(model_path, task, label, out_dir):\n'
             '        result_file = eval_path(out_dir)\n'
             '        if os.path.exists(result_file):\n'
             "            print(f'[SKIP] {label}/{task}: already evaluated.')\n"
             '            r = json.load(open(result_file))\n'
             '            for k, v in r.items():\n'
             "                print(f'  {k}: {v}')\n"
             '            return r\n'
             '\n'
             "        print(f'\\nEvaluating [{label}] on {task} ...')\n"
             "        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
             "        print(f'  Device: {device}')\n"
             '\n'
             "        tok_path = ft_dir(task) if not os.path.exists(os.path.join(model_path, 'tokenizer_config.json')) "
             'else model_path\n'
             '        tok   = AutoTokenizer.from_pretrained(tok_path)\n'
             '        if BASE_REPO_DIR not in sys.path:\n'
             '            sys.path.insert(0, BASE_REPO_DIR)\n'
             "        if 'pruned' in label.lower():\n"
             '            if BASE_REPO_DIR not in sys.path:\n'
             '                sys.path.insert(0, BASE_REPO_DIR)\n'
             '            from models.modeling_bert import CoFiBertForSequenceClassification\n'
             '            from utils.cofi_utils import load_zs, load_model\n'
             '            zs = load_zs(model_path)\n'
             '            if zs is None:\n'
             '                model = CoFiBertForSequenceClassification.from_pretrained(model_path)\n'
             '            else:\n'
             '                model = load_model(model_path, CoFiBertForSequenceClassification, zs)\n'
             '        else:\n'
             '            model = AutoModelForSequenceClassification.from_pretrained(model_path)\n'
             '        n_params = sum(p.numel() for p in model.parameters())\n'
             '        mem_mb   = n_params * 4 / 1e6\n'
             '        model    = model.to(device).eval()\n'
             '\n'
             "        if task == 'mnli':\n"
             "            ds = load_dataset('glue', 'mnli',\n"
             "                              trust_remote_code=True)['validation_matched']\n"
             '        else:\n'
             "            ds = load_dataset('glue', task,\n"
             "                              trust_remote_code=True)['validation']\n"
             '\n'
             '        col_map = {\n'
             "            'sst2': ('sentence',  None),\n"
             "            'qnli': ('question',  'sentence'),\n"
             "            'mnli': ('premise',   'hypothesis'),\n"
             "            'qqp' : ('question1', 'question2'),\n"
             "            'rte' : ('sentence1', 'sentence2'),\n"
             '        }\n'
             '        col_a, col_b = col_map[task]\n'
             '        ds = ds.select(range(min(1000, len(ds))))\n'
             '\n'
             '        def tokenize(batch):\n'
             '            args = (batch[col_a],) if col_b is None else (batch[col_a], batch[col_b])\n'
             "            return tok(*args, padding='max_length', truncation=True,\n"
             '                       max_length=128, return_tensors=None)\n'
             '\n'
             '        ds = ds.map(tokenize, batched=True,\n'
             '                    remove_columns=[c for c in ds.column_names\n'
             "                                    if c not in ['label', 'labels', 'idx']])\n"
             "        ds.set_format('torch')\n"
             '        loader = DataLoader(ds, batch_size=32)\n'
             '\n'
             '        all_preds, all_labels = [], []\n'
             '        total_time, total_examples = 0.0, 0\n'
             '\n'
             '        with torch.no_grad():\n'
             '            for batch in loader:\n'
             "                inp = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', "
             "'token_type_ids']}\n"
             '                model(**inp)\n'
             '                break\n'
             '\n'
             '        with torch.no_grad():\n'
             '            for batch in loader:\n'
             "                labels = batch.pop('label', batch.pop('labels', None))\n"
             '                inp    = {k: v.to(device) for k, v in batch.items()\n'
             "                          if k in ['input_ids', 'attention_mask', 'token_type_ids']}\n"
             '                if torch.cuda.is_available():\n'
             '                    torch.cuda.synchronize()\n'
             '                t0 = time.perf_counter()\n'
             '                out = model(**inp)\n'
             '                if torch.cuda.is_available():\n'
             '                    torch.cuda.synchronize()\n'
             '                t1 = time.perf_counter()\n'
             '                total_time     += (t1 - t0)\n'
             '                total_examples += out.logits.shape[0]\n'
             '                all_preds.extend(out.logits.argmax(-1).cpu().numpy().tolist())\n'
             '                if labels is not None:\n'
             '                    all_labels.extend(labels.cpu().numpy().tolist())\n'
             '\n'
             '        latency_ms = (total_time / total_examples) * 1000\n'
             '        throughput  = total_examples / total_time\n'
             '\n'
             "        if task == 'qqp':\n"
             "            score = hf_evaluate.load('glue', 'qqp').compute(\n"
             "                predictions=all_preds, references=all_labels)['f1']\n"
             "            metric_name = 'F1'\n"
             "        elif task == 'mnli':\n"
             "            score = hf_evaluate.load('glue', 'mnli').compute(\n"
             "                predictions=all_preds, references=all_labels)['accuracy']\n"
             "            metric_name = 'Accuracy'\n"
             '        else:\n'
             "            score = hf_evaluate.load('glue', task).compute(\n"
             "                predictions=all_preds, references=all_labels)['accuracy']\n"
             "            metric_name = 'Accuracy'\n"
             '\n'
             '        # Sparsity excludes embeddings following the paper\n'
             '        # BERT-base embedding params: vocab(30522) + position(512) + token_type(2) = ~23.8M at dim 768\n'
             '        EMBEDDING_PARAMS = (30522 + 512 + 2) * 768\n'
             '        n_non_emb = n_params - EMBEDDING_PARAMS\n'
             '        sparsity_pct = max(0.0, (1 - n_non_emb / BERT_BASE_PARAMS) * 100)\n'
             '\n'
             '        results = {\n'
             "            'label':          label,\n"
             "            'task':           task,\n"
             "            'n_params':       n_params,\n"
             "            'memory_mb':      round(mem_mb, 2),\n"
             "            'latency_ms':     round(latency_ms, 4),\n"
             "            'throughput_eps': round(throughput, 2),\n"
             "            'sparsity_pct':   round(sparsity_pct, 2),\n"
             '            metric_name:      round(score, 4),\n'
             '        }\n'
             '\n'
             '        W = 50\n'
             "        print('=' * W)\n"
             "        print(f'  {label} — {task.upper()}')\n"
             "        print('=' * W)\n"
             "        print(f'  {metric_name:<22}: {score:.4f}')\n"
             "        print(f'  Memory (MB)          : {mem_mb:.1f}')\n"
             "        print(f'  Latency (ms/example) : {latency_ms:.3f}')\n"
             "        print(f'  Throughput (ex/sec)  : {throughput:.1f}')\n"
             "        print(f'  Sparsity %           : {sparsity_pct:.1f}%')\n"
             "        print(f'  Parameters           : {n_params:,}')\n"
             "        print('=' * W)\n"
             '\n'
             '        os.makedirs(out_dir, exist_ok=True)\n'
             "        with open(result_file, 'w') as f:\n"
             '            json.dump(results, f, indent=2)\n'
             "        print(f'  Saved to {result_file}')\n"
             '\n'
             '        del model\n'
             '        if torch.cuda.is_available():\n'
             '            torch.cuda.empty_cache()\n'
             '        return results\n'
             '\n'
             '    for task in tasks:\n'
             "        print(f'\\n--- {task.upper()} unpruned ---')\n"
             "        run_evaluation(ft_dir(task), task, 'BERT-base unpruned', ft_dir(task))\n"
             '\n'
             "        print(f'\\n--- {task.upper()} pruned (60% sparsity) ---')\n"
             "        best = os.path.join(pr_dir(task), 'best')\n"
             '        if not model_saved(best):\n'
             "            print(f'  Pruned model not found at {best}. Run --block 2 --task {task} first.')\n"
             '        else:\n'
             "            run_evaluation(best, task, 'BERT-base pruned 60%', pr_dir(task))\n"
             '\n'
             "    print('\\nBlock 3 done.')\n"
             '\n'
             '# ── Block 4: Results summary ───────────────────────────────────────────────────\n'
             'def block4():\n'
             "    header('BLOCK 4 — Full Results Summary')\n"
             '\n'
             '    metric_label = {\n'
             "        'sst2': 'Accuracy', 'qnli': 'Accuracy', 'mnli': 'Accuracy',\n"
             "        'qqp': 'F1', 'rte': 'Accuracy',\n"
             '    }\n'
             '\n'
             '    def load_result(path):\n'
             '        p = eval_path(path)\n'
             '        if not os.path.exists(p):\n'
             '            return None\n'
             '        return json.load(open(p))\n'
             '\n'
             '    W = 97\n'
             "    print('=' * W)\n"
             "    print('  COFI PRUNING RESULTS — BERT-base on GLUE (60% Sparsity Target)')\n"
             "    print('=' * W)\n"
             '    print(f"  {\'Task\':<6} {\'Model\':<22} {\'Score\':>8} {\'Mem MB\':>9} "\n'
             '          f"{\'Lat ms\':>9} {\'Tput ex/s\':>11} {\'Sparsity\':>10}")\n'
             "    print('-' * W)\n"
             '\n'
             '    for task in ALL_TASKS:\n'
             '        ml  = metric_label[task]\n'
             '        unp = load_result(ft_dir(task))\n'
             '        pru = load_result(pr_dir(task))\n'
             '\n'
             '        def fmt(r):\n'
             '            if r is None:\n'
             "                return ['N/A'] * 5\n"
             "            score = r.get(ml, 'N/A')\n"
             '            return [\n'
             "                f'{score:.4f}' if isinstance(score, float) else str(score),\n"
             "                str(r.get('memory_mb', 'N/A')),\n"
             "                str(r.get('latency_ms', 'N/A')),\n"
             "                str(r.get('throughput_eps', 'N/A')),\n"
             '                f"{r.get(\'sparsity_pct\', \'N/A\')}%",\n'
             '            ]\n'
             '\n'
             '        u = fmt(unp)\n'
             '        p = fmt(pru)\n'
             '        print(f"  {task:<6} {\'Unpruned\':<22} {u[0]:>8} {u[1]:>9} "\n'
             '              f"{u[2]:>9} {u[3]:>11} {u[4]:>10}")\n'
             '        print(f"  {\'\':<6} {\'Pruned 60%\':<22} {p[0]:>8} {p[1]:>9} "\n'
             '              f"{p[2]:>9} {p[3]:>11} {p[4]:>10}")\n'
             '        if unp and pru:\n'
             '            try:\n'
             "                speedup   = unp['latency_ms'] / pru['latency_ms']\n"
             '                retention = float(p[0]) / float(u[0]) * 100\n'
             '                print(f"  {\'\':<6}   -> speedup {speedup:.2f}x   "\n'
             '                      f"score retention {retention:.1f}%")\n'
             '            except Exception:\n'
             '                pass\n'
             "        print('-' * W)\n"
             '\n'
             "    print('=' * W)\n"
             "    print('  Score: Accuracy for SST-2/QNLI/MNLI/RTE, F1 for QQP')\n"
             "    print('  Latency/Throughput: GPU, batch=32, 1000 validation examples')\n"
             "    print('  Sparsity: % reduction vs BERT-base (85M non-embedding params)')\n"
             '\n'
             '# ── Status ─────────────────────────────────────────────────────────────────────\n'
             'def show_status():\n'
             "    header('STATUS')\n"
             '    W = 62\n'
             '    print(f"  {\'\':18}" + \'  \'.join(f\'{t:<6}\' for t in ALL_TASKS))\n'
             "    print('-' * W)\n"
             '    rows = [\n'
             "        ('FT downloaded', [('v' if model_saved(ft_dir(t))\n"
             "                            else '.') for t in ALL_TASKS]),\n"
             "        ('Pruned',        [('v' if model_saved(os.path.join(pr_dir(t), 'best'))\n"
             "                            else '.') for t in ALL_TASKS]),\n"
             "        ('Eval unpruned', [('v' if os.path.exists(eval_path(ft_dir(t)))\n"
             "                            else '.') for t in ALL_TASKS]),\n"
             "        ('Eval pruned',   [('v' if os.path.exists(eval_path(pr_dir(t)))\n"
             "                            else '.') for t in ALL_TASKS]),\n"
             '    ]\n'
             '    for label, row in rows:\n'
             '        print(f"  {label:<18}" + \'  \'.join(f\'{s:<6}\' for s in row))\n'
             "    print('=' * W)\n"
             "    print('  v = done   . = not yet')\n"
             '\n'
             '# ── Main ───────────────────────────────────────────────────────────────────────\n'
             "if __name__ == '__main__':\n"
             "    parser = argparse.ArgumentParser(description='CoFiPruning BERT-base on GLUE')\n"
             "    parser.add_argument('--block', type=int, choices=[0, 1, 2, 3, 4])\n"
             "    parser.add_argument('--task',  type=str, choices=ALL_TASKS)\n"
             "    parser.add_argument('--status', action='store_true')\n"
             '    args = parser.parse_args()\n'
             '\n'
             '    tasks = [args.task] if args.task else ALL_TASKS\n'
             '\n'
             '    if args.status:\n'
             '        show_status()\n'
             '    elif args.block == 0:\n'
             '        block0()\n'
             '    elif args.block == 1:\n'
             '        block1(tasks)\n'
             '    elif args.block == 2:\n'
             '        block2(tasks)\n'
             '    elif args.block == 3:\n'
             '        block3(tasks)\n'
             '    elif args.block == 4:\n'
             '        block4()\n'
             '    else:\n'
             '        parser.print_help()\n',
 'distilbert': '"""\n'
               'CoFiPruning — DistilBERT on 5 GLUE Tasks\n'
               '==========================================\n'
               'Server path: /home/internship2/Cofi_stuff\n'
               'CoFiPruning: /home/internship2/Cofi_stuff/CoFiPruning\n'
               '\n'
               'Usage:\n'
               '  python cofi_distilbert.py --status\n'
               '  python cofi_distilbert.py --block 0\n'
               '  python cofi_distilbert.py --block 1\n'
               '  python cofi_distilbert.py --block 1 --task rte\n'
               '  python cofi_distilbert.py --block 2\n'
               '  python cofi_distilbert.py --block 2 --task rte\n'
               '  python cofi_distilbert.py --block 3\n'
               '  python cofi_distilbert.py --block 3 --task rte\n'
               '  python cofi_distilbert.py --block 4\n'
               '"""\n'
               '\n'
               'import argparse\n'
               'import json\n'
               'import os\n'
               'import re\n'
               'import subprocess\n'
               'import sys\n'
               'import time\n'
               '\n'
               '# ── Paths ──────────────────────────────────────────────────────────────────────\n'
               'BASE_DIR = os.path.dirname(os.path.abspath(__file__))\n'
               "DISTIL_REPO_DIR = os.path.join(BASE_DIR, 'CoFiPruningDistil')\n"
               '\n'
               '# ── MODELING SOURCE ────────────────────────────────────────────────────────\n'
               '# Paste the full content of your ALREADY-PATCHED modeling_distilbert.py here\n'
               '# (the one with del self.ffn, zero-head guards, etc. — not the original draft).\n'
               '# Block 0 writes this string to CoFiPruning/models/modeling_distilbert.py.\n'
               'MODELING_DISTILBERT_SOURCE = """\n'
               '\n'
               '# CoFi-adapted DistilBERT model for sequence classification.\n'
               "# Mirrors the structure of modeling_bert.py but adapted for DistilBERT's architecture:\n"
               '#   - 6 layers (not 12)\n'
               '#   - q_lin/k_lin/v_lin/out_lin instead of query/key/value/dense\n'
               '#   - FFN with lin1/lin2 instead of BertIntermediate/BertOutput\n'
               '#   - No token_type_embeddings\n'
               '#   - No pooler — uses pre_classifier + classifier\n'
               '#   - Config: dim/n_heads/n_layers/hidden_dim instead of hidden_size/num_attention_heads/etc.\n'
               '\n'
               '\n'
               'import logging\n'
               'import math\n'
               'from typing import Optional, Tuple, Union\n'
               '\n'
               'import torch\n'
               'from torch import nn\n'
               'from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss\n'
               'from torch.nn import functional as F\n'
               '\n'
               'from transformers.modeling_outputs import (\n'
               '    BaseModelOutput,\n'
               '    SequenceClassifierOutput,\n'
               ')\n'
               'from transformers.modeling_utils import (\n'
               '    find_pruneable_heads_and_indices,\n'
               '    prune_linear_layer,\n'
               ')\n'
               'from transformers.models.distilbert.modeling_distilbert import (\n'
               '    DistilBertForSequenceClassification,\n'
               '    DistilBertModel,\n'
               '    DistilBertPreTrainedModel,\n'
               '    Embeddings,\n'
               '    FFN,\n'
               '    MultiHeadSelfAttention,\n'
               '    Transformer,\n'
               '    TransformerBlock,\n'
               ')\n'
               'from utils.cofi_utils import *\n'
               '\n'
               'logger = logging.getLogger(__name__)\n'
               '\n'
               '\n'
               '# ── CoFiLayerNorm ──────────────────────────────────────────────────────────────\n'
               'class CoFiLayerNorm(torch.nn.LayerNorm):\n'
               '    # LayerNorm that supports hidden_z masking (same as in modeling_bert.py).\n'
               '\n'
               '    def __init__(self, normalized_shape, eps: float = 1e-5,\n'
               '                 elementwise_affine: bool = True) -> None:\n'
               '        super().__init__(normalized_shape, eps, elementwise_affine)\n'
               '\n'
               '    def forward(self, input, hidden_z=None):\n'
               '        if hidden_z is not None:\n'
               '            remaining_index = torch.where(~hidden_z.eq(0))[0]\n'
               '            compressed_input = torch.index_select(input, dim=-1,\n'
               '                                                   index=remaining_index)\n'
               '            compressed_weight = self.weight[remaining_index]\n'
               '            compressed_bias = self.bias[remaining_index]\n'
               '            normalized_shape = len(remaining_index)\n'
               '            normed_input = F.layer_norm(\n'
               '                compressed_input, [normalized_shape],\n'
               '                compressed_weight, compressed_bias, self.eps)\n'
               '            output = input.clone()\n'
               '            output[:, :, remaining_index] = normed_input\n'
               '        else:\n'
               '            output = F.layer_norm(\n'
               '                input, self.normalized_shape, self.weight, self.bias, self.eps)\n'
               '        return output\n'
               '\n'
               '\n'
               '# ── CoFiDistilBertEmbeddings ───────────────────────────────────────────────────\n'
               'class CoFiDistilBertEmbeddings(Embeddings):\n'
               '    # DistilBert Embeddings with hidden_z masking support.\n'
               '    # DistilBERT has no token_type_embeddings — only word + position.\n'
               "    # Config uses 'dim' not 'hidden_size'.\n"
               '    \n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__(config)\n'
               '        self.LayerNorm = CoFiLayerNorm(config.dim, eps=1e-12)\n'
               '\n'
               '    def forward(self, input_ids=None, input_embeds=None, hidden_z=None):\n'
               '        if input_ids is not None:\n'
               '            input_shape = input_ids.size()\n'
               '            seq_length = input_shape[1]\n'
               '            device = input_ids.device\n'
               '        else:\n'
               '            input_shape = input_embeds.size()[:-1]\n'
               '            seq_length = input_shape[1]\n'
               '            device = input_embeds.device\n'
               '\n'
               '        position_ids = self.position_ids[:, :seq_length]\n'
               '\n'
               '        if input_embeds is None:\n'
               '            input_embeds = self.word_embeddings(input_ids)\n'
               '\n'
               '        position_embeddings = self.position_embeddings(position_ids)\n'
               '        embeddings = input_embeds + position_embeddings\n'
               '\n'
               '        if hidden_z is not None:\n'
               '            embeddings = embeddings.mul(hidden_z)\n'
               '        embeddings = self.LayerNorm(embeddings, hidden_z)\n'
               '        embeddings = self.dropout(embeddings)\n'
               '        if hidden_z is not None:\n'
               '            embeddings = embeddings.mul(hidden_z)\n'
               '        return embeddings\n'
               '\n'
               '\n'
               '# ── CoFiDistilBertSelfAttention ────────────────────────────────────────────────\n'
               'class CoFiDistilBertSelfAttention(MultiHeadSelfAttention):\n'
               '    # MultiHeadSelfAttention with head_z masking support.\n'
               '    # DistilBERT uses q_lin/k_lin/v_lin/out_lin instead of query/key/value/dense.\n'
               '    \n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__(config)\n'
               '        self.attention_head_size = config.dim // config.n_heads\n'
               '\n'
               '    def forward(\n'
               '        self,\n'
               '        query: torch.Tensor,\n'
               '        key: torch.Tensor,\n'
               '        value: torch.Tensor,\n'
               '        mask: torch.Tensor,\n'
               '        head_mask: Optional[torch.Tensor] = None,\n'
               '        output_attentions: bool = False,\n'
               '        head_z: Optional[torch.Tensor] = None,\n'
               '    ) -> Tuple[torch.Tensor, ...]:\n'
               '        if self.v_lin is None or self.n_heads == 0:\n'
               '            return (None, None) if output_attentions else (None,)\n'
               '\n'
               '        bs, q_length, dim = query.size()\n'
               '        k_length = key.size(1)\n'
               '        dim_per_head = self.dim // self.n_heads\n'
               '\n'
               '        mask_reshp = (bs, 1, 1, k_length)\n'
               '\n'
               '        def shape(x: torch.Tensor) -> torch.Tensor:\n'
               '            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)\n'
               '\n'
               '        def unshape(x: torch.Tensor) -> torch.Tensor:\n'
               '            return x.transpose(1, 2).contiguous().view(\n'
               '                bs, -1, self.n_heads * dim_per_head)\n'
               '\n'
               '        q = shape(self.q_lin(query))\n'
               '        k = shape(self.k_lin(key))\n'
               '        v = shape(self.v_lin(value))\n'
               '\n'
               '        q = q / math.sqrt(dim_per_head)\n'
               '        scores = torch.matmul(q, k.transpose(2, 3))\n'
               '        mask = (mask == 0).view(mask_reshp).expand_as(scores)\n'
               '        scores = scores.masked_fill(\n'
               '            mask, torch.tensor(torch.finfo(scores.dtype).min))\n'
               '\n'
               '        weights = nn.functional.softmax(scores, dim=-1)\n'
               '        weights = self.dropout(weights)\n'
               '\n'
               '        if head_mask is not None:\n'
               '            weights = weights * head_mask\n'
               '\n'
               '        context = torch.matmul(weights, v)\n'
               '\n'
               '        # Apply head_z mask (CoFi-specific)\n'
               '        if head_z is not None:\n'
               '            context *= head_z\n'
               '\n'
               '        context = unshape(context)\n'
               '        context = self.out_lin(context)\n'
               '\n'
               '        if output_attentions:\n'
               '            return (context, weights)\n'
               '        else:\n'
               '            return (context,)\n'
               '\n'
               '\n'
               '# ── CoFiDistilBertAttentionOutput (replaces BertSelfOutput role) ───────────────\n'
               'class CoFiDistilBertAttentionOutput(nn.Module):\n'
               '    # Handles the post-attention dense + LayerNorm + residual.\n'
               "    # In DistilBERT this is part of TransformerBlock's sa_layer_norm.\n"
               "    # We split it out to match CoFi's head_layer_z and hidden_z masking pattern.\n"
               '    \n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        # out_lin is already in MultiHeadSelfAttention; here we handle the norm\n'
               '        self.LayerNorm = CoFiLayerNorm(config.dim, eps=1e-12)\n'
               '        self.dropout = nn.Dropout(config.dropout)\n'
               '        self.config = config\n'
               '\n'
               '    def forward(self, hidden_states, input_tensor,\n'
               '                head_layer_z=None, hidden_z=None, inference=False):\n'
               '        if hidden_states is None:\n'
               '            return input_tensor\n'
               '        if head_layer_z is not None:\n'
               '            hidden_states = hidden_states.mul(head_layer_z)\n'
               '        if not inference and hidden_states.sum().eq(0).item():\n'
               '            hidden_states = hidden_states + input_tensor\n'
               '        else:\n'
               '            if hidden_z is not None:\n'
               '                hidden_states = hidden_states.mul(hidden_z)\n'
               '            hidden_states = self.dropout(hidden_states)\n'
               '            hidden_states = self.LayerNorm(hidden_states + input_tensor, hidden_z)\n'
               '            if hidden_z is not None:\n'
               '                hidden_states = hidden_states.mul(hidden_z)\n'
               '        return hidden_states\n'
               '\n'
               '\n'
               '# ── CoFiDistilBertFFNOutput ────────────────────────────────────────────────────\n'
               'class CoFiDistilBertFFNOutput(nn.Module):\n'
               '    # Handles the post-FFN LayerNorm + residual with mlp_z and hidden_z masking.\n'
               "    # DistilBERT's FFN uses lin1 (up) and lin2 (down) — equivalent to\n"
               '    # BertIntermediate + BertOutput in BERT.\n'
               '\n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.LayerNorm = CoFiLayerNorm(config.dim, eps=1e-12)\n'
               '        self.dropout = nn.Dropout(config.dropout)\n'
               '        self.config = config\n'
               '\n'
               '    def forward(self, hidden_states, input_tensor, mlp_z,\n'
               '                hidden_z=None, inference=False):\n'
               '        if mlp_z is not None:\n'
               '            hidden_states = hidden_states * mlp_z\n'
               '        if not inference and hidden_states.sum().eq(0).item():\n'
               '            return hidden_states + input_tensor\n'
               '        else:\n'
               '            if hidden_z is not None:\n'
               '                hidden_states = hidden_states.mul(hidden_z)\n'
               '            hidden_states = self.dropout(hidden_states)\n'
               '            hidden_states = self.LayerNorm(hidden_states + input_tensor, hidden_z)\n'
               '            if hidden_z is not None:\n'
               '                hidden_states = hidden_states.mul(hidden_z)\n'
               '        return hidden_states\n'
               '\n'
               '\n'
               '# ── CoFiDistilBertTransformerBlock ────────────────────────────────────────────\n'
               'class CoFiDistilBertTransformerBlock(TransformerBlock):\n'
               '    # TransformerBlock with full CoFi mask support.\n'
               "    # Replaces TransformerBlock's attention and LayerNorms with CoFi versions.\n"
               '    \n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__(config)\n'
               '        self.attention = CoFiDistilBertSelfAttention(config)\n'
               '        self.attn_output = CoFiDistilBertAttentionOutput(config)\n'
               '        self.ffn_output = CoFiDistilBertFFNOutput(config)\n'
               '        # Move lin1/lin2 out of self.ffn (avoid shared-tensor save error)\n'
               '        self.ffn_lin1 = self.ffn.lin1\n'
               '        self.ffn_lin2 = self.ffn.lin2\n'
               '        self.ffn_activation = self.ffn.activation\n'
               '        del self.ffn\n'
               '        self.config = config\n'
               '\n'
               '    def forward(\n'
               '        self,\n'
               '        x: torch.Tensor,\n'
               '        attn_mask: Optional[torch.Tensor] = None,\n'
               '        head_mask: Optional[torch.Tensor] = None,\n'
               '        output_attentions: bool = False,\n'
               '        head_z: Optional[torch.Tensor] = None,\n'
               '        head_layer_z: Optional[torch.Tensor] = None,\n'
               '        intermediate_z: Optional[torch.Tensor] = None,\n'
               '        mlp_z: Optional[torch.Tensor] = None,\n'
               '        hidden_z: Optional[torch.Tensor] = None,\n'
               '    ) -> Tuple[torch.Tensor, ...]:\n'
               '\n'
               '        # ── Self-Attention ──────────────────────────────────────────────────\n'
               '        sa_output = self.attention(\n'
               '            query=x, key=x, value=x,\n'
               '            mask=attn_mask,\n'
               '            head_mask=head_mask,\n'
               '            output_attentions=output_attentions,\n'
               '            head_z=head_z,\n'
               '        )\n'
               '        if output_attentions:\n'
               '            sa_context, sa_weights = sa_output\n'
               '        else:\n'
               '            sa_context = sa_output[0]\n'
               '\n'
               '        # Post-attention: head_layer_z masking + LayerNorm + residual\n'
               '        sa_output_normed = self.attn_output(\n'
               '            sa_context, x, head_layer_z=head_layer_z, hidden_z=hidden_z)\n'
               '\n'
               '        # ── FFN ─────────────────────────────────────────────────────────────\n'
               '        if self.ffn_lin1 is None:\n'
               '            ffn_output = sa_output_normed\n'
               '        else:\n'
               '            # lin1 (up-projection)\n'
               '            intermediate = self.ffn_lin1(sa_output_normed)\n'
               '            intermediate = self.ffn_activation(intermediate)\n'
               '            # intermediate_z masking\n'
               '            if intermediate_z is not None:\n'
               '                intermediate = intermediate.mul(intermediate_z)\n'
               '            # lin2 (down-projection)\n'
               '            ffn_out = self.ffn_lin2(intermediate)\n'
               '\n'
               '            # Post-FFN: mlp_z masking + LayerNorm + residual\n'
               '            ffn_output = self.ffn_output(\n'
               '                ffn_out, sa_output_normed, mlp_z=mlp_z, hidden_z=hidden_z)\n'
               '\n'
               '        output = (ffn_output,)\n'
               '        if output_attentions:\n'
               '            output = (sa_weights,) + output\n'
               '        # append attention output for layer distillation (matches BERT convention)\n'
               '        output = output + (sa_output_normed,)\n'
               '        return output\n'
               '\n'
               '    def prune_heads(self, heads):\n'
               '        if len(heads) == 0:\n'
               '            return\n'
               '        heads, index = find_pruneable_heads_and_indices(\n'
               '            heads,\n'
               '            self.attention.n_heads,\n'
               '            self.attention.attention_head_size,\n'
               '            self.attention.pruned_heads,\n'
               '        )\n'
               '        if len(index) == 0:\n'
               '            self.attention.q_lin = None\n'
               '            self.attention.k_lin = None\n'
               '            self.attention.v_lin = None\n'
               '            self.attention.out_lin = None\n'
               '        else:\n'
               '            self.attention.q_lin = prune_linear_layer(\n'
               '                self.attention.q_lin, index)\n'
               '            self.attention.k_lin = prune_linear_layer(\n'
               '                self.attention.k_lin, index)\n'
               '            self.attention.v_lin = prune_linear_layer(\n'
               '                self.attention.v_lin, index)\n'
               '            self.attention.out_lin = prune_linear_layer(\n'
               '                self.attention.out_lin, index, dim=1)\n'
               '        self.attention.n_heads = self.attention.n_heads - len(heads)\n'
               '        self.attention.dim = (self.attention.attention_head_size\n'
               '                              * self.attention.n_heads)\n'
               '        self.attention.pruned_heads = self.attention.pruned_heads.union(heads)\n'
               '\n'
               '\n'
               '# ── CoFiDistilBertTransformer ──────────────────────────────────────────────────\n'
               'class CoFiDistilBertTransformer(Transformer):\n'
               '    # Transformer stack that threads CoFi mask variables per layer.\n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__(config)\n'
               '        self.layer = nn.ModuleList(\n'
               '            [CoFiDistilBertTransformerBlock(config)\n'
               '             for _ in range(config.n_layers)])\n'
               '\n'
               '    def forward(\n'
               '        self,\n'
               '        x: torch.Tensor,\n'
               '        attn_mask: Optional[torch.Tensor] = None,\n'
               '        head_mask: Optional[torch.Tensor] = None,\n'
               '        output_attentions: bool = False,\n'
               '        output_hidden_states: bool = False,\n'
               '        return_dict: Optional[bool] = None,\n'
               '        head_z: Optional[torch.Tensor] = None,\n'
               '        head_layer_z: Optional[torch.Tensor] = None,\n'
               '        intermediate_z: Optional[torch.Tensor] = None,\n'
               '        mlp_z: Optional[torch.Tensor] = None,\n'
               '        hidden_z: Optional[torch.Tensor] = None,\n'
               '    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:\n'
               '\n'
               '        all_hidden_states = () if output_hidden_states else None\n'
               '        all_attentions = () if output_attentions else None\n'
               '\n'
               '        hidden_state = x\n'
               '        for i, layer_module in enumerate(self.layer):\n'
               '            if output_hidden_states:\n'
               '                all_hidden_states = all_hidden_states + (hidden_state,)\n'
               '\n'
               '            layer_outputs = layer_module(\n'
               '                hidden_state,\n'
               '                attn_mask=attn_mask,\n'
               '                head_mask=head_mask[i] if head_mask is not None else None,\n'
               '                output_attentions=output_attentions,\n'
               '                head_z=head_z[i] if head_z is not None else None,\n'
               '                head_layer_z=head_layer_z[i] if head_layer_z is not None else None,\n'
               '                intermediate_z=intermediate_z[i] if intermediate_z is not None else None,\n'
               '                mlp_z=mlp_z[i] if mlp_z is not None else None,\n'
               '                hidden_z=hidden_z,\n'
               '            )\n'
               '            hidden_state = layer_outputs[0] if not output_attentions else layer_outputs[1]\n'
               '\n'
               '            if output_attentions:\n'
               '                all_attentions = all_attentions + (layer_outputs[0],)\n'
               '\n'
               '        if output_hidden_states:\n'
               '            all_hidden_states = all_hidden_states + (hidden_state,)\n'
               '\n'
               '        if not return_dict:\n'
               '            return tuple(\n'
               '                v for v in [hidden_state, all_hidden_states, all_attentions]\n'
               '                if v is not None)\n'
               '        return BaseModelOutput(\n'
               '            last_hidden_state=hidden_state,\n'
               '            hidden_states=all_hidden_states,\n'
               '            attentions=all_attentions,\n'
               '        )\n'
               '\n'
               '\n'
               '# ── CoFiDistilBertModel ────────────────────────────────────────────────────────\n'
               'class CoFiDistilBertModel(DistilBertModel):\n'
               '    # DistilBertModel with CoFi mask variables threaded through.\n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__(config)\n'
               '        self.embeddings = CoFiDistilBertEmbeddings(config)\n'
               '        self.transformer = CoFiDistilBertTransformer(config)\n'
               '\n'
               '    def forward(\n'
               '        self,\n'
               '        input_ids=None,\n'
               '        attention_mask=None,\n'
               '        head_mask=None,\n'
               '        inputs_embeds=None,\n'
               '        output_attentions=None,\n'
               '        output_hidden_states=None,\n'
               '        return_dict=None,\n'
               '        head_z=None,\n'
               '        head_layer_z=None,\n'
               '        intermediate_z=None,\n'
               '        mlp_z=None,\n'
               '        hidden_z=None,\n'
               '    ):\n'
               '        output_attentions = (output_attentions if output_attentions is not None\n'
               '                             else self.config.output_attentions)\n'
               '        output_hidden_states = (output_hidden_states\n'
               '                                if output_hidden_states is not None\n'
               '                                else self.config.output_hidden_states)\n'
               '        return_dict = (return_dict if return_dict is not None\n'
               '                       else self.config.use_return_dict)\n'
               '\n'
               '        if input_ids is not None and inputs_embeds is not None:\n'
               '            raise ValueError(\n'
               '                "You cannot specify both input_ids and inputs_embeds at the same time")\n'
               '        elif input_ids is not None:\n'
               '            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)\n'
               '            input_shape = input_ids.size()\n'
               '        elif inputs_embeds is not None:\n'
               '            input_shape = inputs_embeds.size()[:-1]\n'
               '        else:\n'
               '            raise ValueError(\n'
               '                "You have to specify either input_ids or inputs_embeds")\n'
               '\n'
               '        device = input_ids.device if input_ids is not None else inputs_embeds.device\n'
               '\n'
               '        if attention_mask is None:\n'
               '            attention_mask = torch.ones(input_shape, device=device)\n'
               '\n'
               '        head_mask = self.get_head_mask(head_mask, self.config.n_layers)\n'
               '\n'
               '        embeddings = self.embeddings(\n'
               '            input_ids=input_ids,\n'
               '            input_embeds=inputs_embeds,\n'
               '            hidden_z=hidden_z,\n'
               '        )\n'
               '\n'
               '        transformer_output = self.transformer(\n'
               '            x=embeddings,\n'
               '            attn_mask=attention_mask,\n'
               '            head_mask=head_mask,\n'
               '            output_attentions=output_attentions,\n'
               '            output_hidden_states=output_hidden_states,\n'
               '            return_dict=return_dict,\n'
               '            head_z=head_z,\n'
               '            head_layer_z=head_layer_z,\n'
               '            intermediate_z=intermediate_z,\n'
               '            mlp_z=mlp_z,\n'
               '            hidden_z=hidden_z,\n'
               '        )\n'
               '\n'
               '        return transformer_output\n'
               '\n'
               '\n'
               '# ── CoFiDistilBertForSequenceClassification ────────────────────────────────────\n'
               'class CoFiDistilBertForSequenceClassification(DistilBertForSequenceClassification):\n'
               '\n'
               '    # CoFi-prunable DistilBERT for sequence classification.\n'
               '    # Mirrors CoFiBertForSequenceClassification from modeling_bert.py.\n'
               '\n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__(config)\n'
               '        self.distilbert = CoFiDistilBertModel(config)\n'
               '\n'
               '        self.do_layer_distill = getattr(config, "do_layer_distill", False)\n'
               '        if self.do_layer_distill:\n'
               '            self.layer_transformation = nn.Linear(config.dim, config.dim)\n'
               '        else:\n'
               '            self.layer_transformation = None\n'
               '\n'
               '    def forward(\n'
               '        self,\n'
               '        input_ids=None,\n'
               '        attention_mask=None,\n'
               '        head_mask=None,\n'
               '        inputs_embeds=None,\n'
               '        labels=None,\n'
               '        output_attentions=None,\n'
               '        output_hidden_states=None,\n'
               '        return_dict=None,\n'
               '        head_z=None,\n'
               '        head_layer_z=None,\n'
               '        intermediate_z=None,\n'
               '        mlp_z=None,\n'
               '        hidden_z=None,\n'
               '    ):\n'
               '        return_dict = (return_dict if return_dict is not None\n'
               '                       else self.config.use_return_dict)\n'
               '\n'
               '        outputs = self.distilbert(\n'
               '            input_ids=input_ids,\n'
               '            attention_mask=attention_mask,\n'
               '            head_mask=head_mask,\n'
               '            inputs_embeds=inputs_embeds,\n'
               '            output_attentions=output_attentions,\n'
               '            output_hidden_states=output_hidden_states,\n'
               '            return_dict=return_dict,\n'
               '            head_z=head_z,\n'
               '            head_layer_z=head_layer_z,\n'
               '            intermediate_z=intermediate_z,\n'
               '            mlp_z=mlp_z,\n'
               '            hidden_z=hidden_z,\n'
               '        )\n'
               '\n'
               '        hidden_state = outputs[0]      # (bs, seq_len, dim)\n'
               '        pooled_output = hidden_state[:, 0]  # CLS token\n'
               '        pooled_output = self.pre_classifier(pooled_output)\n'
               '        pooled_output = nn.ReLU()(pooled_output)\n'
               '        pooled_output = self.dropout(pooled_output)\n'
               '        logits = self.classifier(pooled_output)\n'
               '\n'
               '        loss = None\n'
               '        if labels is not None:\n'
               '            if self.config.problem_type is None:\n'
               '                if self.num_labels == 1:\n'
               '                    self.config.problem_type = "regression"\n'
               '                elif self.num_labels > 1 and (\n'
               '                        labels.dtype == torch.long or labels.dtype == torch.int):\n'
               '                    self.config.problem_type = "single_label_classification"\n'
               '                else:\n'
               '                    self.config.problem_type = "multi_label_classification"\n'
               '\n'
               '            if self.config.problem_type == "regression":\n'
               '                loss_fct = MSELoss()\n'
               '                loss = (loss_fct(logits.squeeze(), labels.squeeze())\n'
               '                        if self.num_labels == 1\n'
               '                        else loss_fct(logits, labels))\n'
               '            elif self.config.problem_type == "single_label_classification":\n'
               '                loss_fct = CrossEntropyLoss()\n'
               '                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))\n'
               '            elif self.config.problem_type == "multi_label_classification":\n'
               '                loss_fct = BCEWithLogitsLoss()\n'
               '                loss = loss_fct(logits, labels)\n'
               '\n'
               '        if not return_dict:\n'
               '            output = (logits,) + outputs[1:]\n'
               '            return ((loss,) + output) if loss is not None else output\n'
               '\n'
               '        return SequenceClassifierOutput(\n'
               '            loss=loss,\n'
               '            logits=logits,\n'
               '            hidden_states=outputs.hidden_states,\n'
               '            attentions=outputs.attentions,\n'
               '        )\n'
               '\n'
               '"""\n'
               '\n'
               '# ── Constants ──────────────────────────────────────────────────────────────────\n'
               'SPARSITY    = 0.6\n'
               'SEED        = 57\n'
               "ALL_TASKS   = ['sst2', 'qnli', 'mnli', 'qqp', 'rte']\n"
               "SMALL_TASKS = {'rte', 'mrpc', 'cola', 'stsb'}\n"
               '\n'
               '# Pre-finetuned DistilBERT models — fill in HuggingFace model IDs as you find them\n'
               'DISTIL_PRETRAINED_FT = {\n'
               "    'sst2': 'distilbert-base-uncased-finetuned-sst-2-english', \n"
               "    'qnli': 'PStingley/distilbert-base-uncased-finetuned-qnli',  \n"
               "    'mnli': 'dzungever/distilbert-finetuned-mnli',  \n"
               "    'qqp' : 'gokulsrinivasagan/distilbert-base-uncased_qqp',   \n"
               "    'rte' : 'textattack/distilbert-base-uncased-RTE',   \n"
               '}\n'
               '\n'
               '# DistilBERT non-embedding param count (6 layers, dim=768, n_heads=12, hidden_dim=3072)\n'
               '# = 6 * (768*768*4 + 768*4 + 768*3072*2 + 768) = ~52.3M non-embedding params\n'
               '# We use 52_259_328 as baseline for sparsity calculation\n'
               'DISTILBERT_BASE_PARAMS = 42_527_232  # calculate_parameters() on unpruned DistilBERT-base '
               '(non-embedding)\n'
               '# DistilBERT embedding params: vocab(30522) + position(512) at dim=768\n'
               'DISTILBERT_EMBEDDING_PARAMS = 23_835_648  # sum(model.distilbert.embeddings.parameters()) — includes '
               'embeddings LayerNorm weight+bias + 2 * 768  # + embeddings LayerNorm weight+bias\n'
               '\n'
               '# ── Helpers ────────────────────────────────────────────────────────────────────\n'
               'def ft_dir(task):\n'
               "    return os.path.join(BASE_DIR, f'ft_distilbert_{task}')\n"
               '\n'
               'def pr_dir(task):\n'
               "    return os.path.join(BASE_DIR, f'pr_distilbert_{task}_s{int(SPARSITY*100)}')\n"
               '\n'
               'def eval_path(d):\n'
               "    return os.path.join(d, 'eval_results.json')\n"
               '\n'
               'def model_saved(path):\n'
               "    return (os.path.exists(os.path.join(path, 'pytorch_model.bin')) or\n"
               "            os.path.exists(os.path.join(path, 'model.safetensors')))\n"
               '\n'
               'def task_cfg(task):\n'
               '    if task in SMALL_TASKS:\n'
               '        return dict(prune_epochs=100, eval_steps=50, save_steps=50,\n'
               '                    prepruning=4, lag_warmup=20, layer_distill_v=4)\n'
               '    return dict(prune_epochs=20, eval_steps=500, save_steps=500,\n'
               '                prepruning=1, lag_warmup=2, layer_distill_v=3)\n'
               '\n'
               'def header(msg):\n'
               '    W = 60\n'
               "    print('\\n' + '=' * W)\n"
               "    print(f'  {msg}')\n"
               "    print('=' * W)\n"
               '\n'
               'def patch_file(fpath, old, new, description):\n'
               '    txt = open(fpath).read()\n'
               '    if new and new in txt:\n'
               "        print(f'  [already patched] {description}')\n"
               '        return\n'
               '    if old not in txt:\n'
               "        print(f'  [not found]       {description}')\n"
               '        return\n'
               "    open(fpath, 'w').write(txt.replace(old, new))\n"
               "    print(f'  [patched]         {description}')\n"
               '\n'
               '# ── Block 0: Fresh clone + all patches + write modeling file ──────────────────\n'
               'def block0():\n'
               "    header('BLOCK 0 — Fresh clone + patch CoFiPruning for DistilBERT')\n"
               '\n'
               "    if MODELING_DISTILBERT_SOURCE.strip() == 'PASTE THE FULL CONTENT OF modeling_distilbert.py HERE':\n"
               "        print('ERROR: MODELING_DISTILBERT_SOURCE is still a placeholder.')\n"
               "        print('Paste the content of modeling_distilbert.py into the string in this script.')\n"
               '        sys.exit(1)\n'
               '\n'
               '    # ── Fresh clone ────────────────────────────────────────────────────────────\n'
               "    print('\\n  Refreshing repo...')\n"
               '    if os.path.exists(DISTIL_REPO_DIR):\n'
               "        subprocess.run(['chmod', '-R', 'u+w', DISTIL_REPO_DIR], check=True)\n"
               "        subprocess.run(['rm', '-rf', DISTIL_REPO_DIR], check=True)\n"
               "        print(f'  Removed old {DISTIL_REPO_DIR}')\n"
               '    subprocess.run(\n'
               "        ['git', 'clone', 'https://github.com/princeton-nlp/CoFiPruning.git', DISTIL_REPO_DIR],\n"
               '        check=True)\n'
               "    print(f'  Cloned fresh into {DISTIL_REPO_DIR}')\n"
               '\n'
               '    # ── Write modeling_distilbert.py (already fully patched — no further edits) ─\n'
               "    modeling_dst = os.path.join(DISTIL_REPO_DIR, 'models', 'modeling_distilbert.py')\n"
               "    with open(modeling_dst, 'w') as f:\n"
               '        f.write(MODELING_DISTILBERT_SOURCE)\n'
               "    print(f'  [written]         models/modeling_distilbert.py '\n"
               "          f'({len(MODELING_DISTILBERT_SOURCE.splitlines())} lines)')\n"
               '\n'
               '    # ── Standard import patches (all .py files) ────────────────────────────────\n'
               "    print('\\n  Standard import patches...')\n"
               '    IMPORT_PATCHES = [\n'
               "        ('from transformers.file_utils import hf_bucket_url, cached_path',\n"
               "         'from huggingface_hub import cached_download as cached_path'),\n"
               "        ('from transformers.file_utils import cached_path',\n"
               "         'from huggingface_hub import cached_download as cached_path'),\n"
               "        ('from transformers.file_utils import hf_bucket_url', ''),\n"
               "        ('from datasets import load_dataset, load_metric, DatasetDict',\n"
               "         'from datasets import load_dataset, DatasetDict\\nimport evaluate'),\n"
               '        (\'metric = load_metric("glue", data_args.task_name)\',\n'
               '         \'metric = evaluate.load("glue", data_args.task_name)\'),\n'
               '        (\'metric = load_metric("accuracy")\',\n'
               '         \'metric = evaluate.load("accuracy")\'),\n'
               "        ('from black import main', ''),\n"
               '    ]\n'
               '    for root, dirs, files in os.walk(DISTIL_REPO_DIR):\n'
               "        dirs[:] = [d for d in dirs if d != '.git']\n"
               '        for fname in files:\n'
               "            if not fname.endswith('.py'):\n"
               '                continue\n'
               '            fpath = os.path.join(root, fname)\n'
               '            rel   = os.path.relpath(fpath, DISTIL_REPO_DIR)\n'
               '            for old, new in IMPORT_PATCHES:\n'
               '                patch_file(fpath, old, new, rel)\n'
               '\n'
               '    # ── run_glue_prune.py ──────────────────────────────────────────────────────\n'
               "    print('\\n  Patching run_glue_prune.py...')\n"
               "    glue_path = os.path.join(DISTIL_REPO_DIR, 'run_glue_prune.py')\n"
               '\n'
               '    patch_file(glue_path,\n'
               '        \'load_dataset("glue", data_args.task_name)\',\n'
               '        \'load_dataset("glue", data_args.task_name, trust_remote_code=True)\',\n'
               "        'add trust_remote_code')\n"
               '    patch_file(glue_path,\n'
               '        \'"evaluation_strategy"\',\n'
               '        \'"eval_strategy"\',\n'
               "        'evaluation_strategy -> eval_strategy')\n"
               '\n'
               '    patch_file(glue_path,\n'
               "        'from models.modeling_bert import CoFiBertForSequenceClassification\\n'\n"
               "        'from models.modeling_roberta import CoFiRobertaForSequenceClassification',\n"
               "        'from models.modeling_bert import CoFiBertForSequenceClassification\\n'\n"
               "        'from models.modeling_roberta import CoFiRobertaForSequenceClassification\\n'\n"
               "        'from models.modeling_distilbert import CoFiDistilBertForSequenceClassification',\n"
               "        'run_glue_prune.py: import CoFiDistilBertForSequenceClassification')\n"
               '\n'
               '    # Model selection — distilbert checked FIRST to avoid "bert-base-uncased"\n'
               '    # substring match (e.g. "bert-base-uncased" in "distilbert-base-uncased" is True),\n'
               '    # and local finetune dirs (no recognizable prefix) fall back to CoFiBert.\n'
               '    patch_file(glue_path,\n'
               "        '    Model = CoFiBertForSequenceClassification if "
               "model_args.model_name_or_path.startswith(\\n'\n"
               '        \'        "bert") else CoFiRobertaForSequenceClassification\',\n'
               '        \'    if "distilbert" in model_args.model_name_or_path.lower():\\n\'\n'
               "        '        Model = CoFiDistilBertForSequenceClassification\\n'\n"
               '        \'    elif (model_args.model_name_or_path.startswith("bert")\\n\'\n'
               "        '          or os.path.exists(os.path.join(model_args.model_name_or_path, "
               '"config.json"))):\\n\'\n'
               "        '        Model = CoFiBertForSequenceClassification\\n'\n"
               "        '    else:\\n'\n"
               "        '        Model = CoFiRobertaForSequenceClassification',\n"
               "        'run_glue_prune.py: model selection — distilbert-first, local-path-aware')\n"
               '\n'
               '    # ── models/modeling_bert.py: remove broken from_pretrained override ────────\n'
               "    print('\\n  Patching models/modeling_bert.py...')\n"
               "    bert_path = os.path.join(DISTIL_REPO_DIR, 'models', 'modeling_bert.py')\n"
               '    src = open(bert_path).read()\n'
               '    pat = re.compile(\n'
               "        r'[ \\t]*@classmethod\\s*\\n[ \\t]*def from_pretrained\\(cls.*?(?=\\n[ \\t]{0,4}(?:def |class "
               "|\\Z))',\n"
               '        re.DOTALL)\n'
               '    if re.search(pat, src):\n'
               "        open(bert_path, 'w').write(re.sub(pat, '', src))\n"
               "        print('  [patched]         modeling_bert.py: removed from_pretrained override')\n"
               '    else:\n'
               "        print('  [already patched] modeling_bert.py')\n"
               '\n'
               '    # ── utils/cofi_utils.py ────────────────────────────────────────────────────\n'
               "    print('\\n  Patching utils/cofi_utils.py...')\n"
               "    utils_path = os.path.join(DISTIL_REPO_DIR, 'utils', 'cofi_utils.py')\n"
               '\n'
               '    # safetensors support\n'
               '    patch_file(utils_path,\n'
               '        \'    p = os.path.join(model_path, "pytorch_model.bin")\\n\'\n'
               '        \'    loaded_weights = torch.load(p, map_location="cpu")\',\n'
               '        \'    p_bin  = os.path.join(model_path, "pytorch_model.bin")\\n\'\n'
               '        \'    p_safe = os.path.join(model_path, "model.safetensors")\\n\'\n'
               "        '    if os.path.exists(p_bin):\\n'\n"
               '        \'        loaded_weights = torch.load(p_bin, map_location="cpu")\\n\'\n'
               "        '    elif os.path.exists(p_safe):\\n'\n"
               "        '        from safetensors.torch import load_file\\n'\n"
               "        '        loaded_weights = load_file(p_safe)\\n'\n"
               "        '    else:\\n'\n"
               '        \'        raise FileNotFoundError(f"No weights found in {model_path}")\',\n'
               "        'cofi_utils: safetensors support')\n"
               '\n'
               '    # _get_layers helper\n'
               '    patch_file(utils_path,\n'
               "        'def prune_model_with_z(zs, model):',\n"
               "        'def _get_layers(bert):\\n'\n"
               '        \'    """Return list of transformer layer objects, model-agnostically."""\\n\'\n'
               '        \'    if hasattr(bert, "encoder"):\\n\'\n'
               "        '        return bert.encoder.layer\\n'\n"
               '        \'    elif hasattr(bert, "transformer"):\\n\'\n'
               "        '        return bert.transformer.layer\\n'\n"
               "        '    else:\\n'\n"
               '        \'        raise ValueError(f"Cannot find layers in {type(bert)}")\\n\\n\'\n'
               "        'def prune_model_with_z(zs, model):',\n"
               "        'cofi_utils: _get_layers helper')\n"
               '\n'
               '    # prune_model_with_z — model type + layer count detection\n'
               '    patch_file(utils_path,\n'
               "        'def prune_model_with_z(zs, model):\\n'\n"
               "        '    if zs is None:\\n'\n"
               "        '        return None, None\\n'\n"
               '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\',\n'
               "        'def prune_model_with_z(zs, model):\\n'\n"
               "        '    if zs is None:\\n'\n"
               "        '        return None, None\\n'\n"
               '        \'    if hasattr(model, "bert"):\\n\'\n'
               "        '        bert = model.bert\\n'\n"
               "        '        num_layers = model.config.num_hidden_layers\\n'\n"
               '        \'    elif hasattr(model, "roberta"):\\n\'\n'
               "        '        bert = model.roberta\\n'\n"
               "        '        num_layers = model.config.num_hidden_layers\\n'\n"
               '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
               "        '        bert = model.distilbert\\n'\n"
               "        '        num_layers = model.config.n_layers\\n'\n"
               "        '    else:\\n'\n"
               '        \'        raise ValueError(f"Unknown model type: {type(model)}")\',\n'
               "        'cofi_utils: prune_model_with_z — model type detection')\n"
               '\n'
               '    # hidden_z layer loop — bert/distilbert, model-agnostic + zero-head safe\n'
               '    patch_file(utils_path,\n'
               "        '        for layer in range(0, 12):\\n'\n"
               "        '            if bert.encoder.layer[layer].attention.self.query is not None:\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.query = \\\\\\n'\n"
               "        '                    prune_layer(bert.encoder.layer[layer].attention.self.query , index, "
               "dim=1)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.key = \\\\\\n'\n"
               "        '                    prune_layer(bert.encoder.layer[layer].attention.self.key , index, "
               "dim=1)\\n'\n"
               "        '            if bert.encoder.layer[layer].attention.self.value is not None:\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.value = \\\\\\n'\n"
               "        '                    prune_layer(bert.encoder.layer[layer].attention.self.value , index, "
               "dim=1)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.output.dense = \\\\\\n'\n"
               "        '                    prune_layer(bert.encoder.layer[layer].attention.output.dense , index, "
               "dim=0)\\n'\n"
               "        '                prune_layer_norm(bert.encoder.layer[layer].attention.output.LayerNorm, "
               "index)\\n'\n"
               "        '            if bert.encoder.layer[layer].intermediate.dense is not None:\\n'\n"
               "        '                bert.encoder.layer[layer].intermediate.dense = \\\\\\n'\n"
               "        '                    prune_layer( bert.encoder.layer[layer].intermediate.dense, index, "
               "dim=1)\\n'\n"
               "        '                bert.encoder.layer[layer].output.dense = \\\\\\n'\n"
               "        '                    prune_layer( bert.encoder.layer[layer].output.dense, index, dim=0)\\n'\n"
               "        '                prune_layer_norm(bert.encoder.layer[layer].output.LayerNorm, index)',\n"
               "        '        layers = _get_layers(bert)\\n'\n"
               "        '        for layer in range(num_layers):\\n'\n"
               "        '            lyr = layers[layer]\\n'\n"
               '        \'            if hasattr(lyr, "attention") and hasattr(lyr.attention, "self"):\\n\'\n'
               "        '                if lyr.attention.self.query is not None:\\n'\n"
               "        '                    lyr.attention.self.query = prune_layer(lyr.attention.self.query, index, "
               "dim=1)\\n'\n"
               "        '                    lyr.attention.self.key = prune_layer(lyr.attention.self.key, index, "
               "dim=1)\\n'\n"
               "        '                if lyr.attention.self.value is not None:\\n'\n"
               "        '                    lyr.attention.self.value = prune_layer(lyr.attention.self.value, index, "
               "dim=1)\\n'\n"
               "        '                    lyr.attention.output.dense = prune_layer(lyr.attention.output.dense, "
               "index, dim=0)\\n'\n"
               "        '                    prune_layer_norm(lyr.attention.output.LayerNorm, index)\\n'\n"
               "        '                if lyr.intermediate.dense is not None:\\n'\n"
               "        '                    lyr.intermediate.dense = prune_layer(lyr.intermediate.dense, index, "
               "dim=1)\\n'\n"
               "        '                    lyr.output.dense = prune_layer(lyr.output.dense, index, dim=0)\\n'\n"
               "        '                    prune_layer_norm(lyr.output.LayerNorm, index)\\n'\n"
               '        \'            elif hasattr(lyr, "attention") and hasattr(lyr.attention, "q_lin"):\\n\'\n'
               "        '                if lyr.attention.q_lin is not None:\\n'\n"
               "        '                    lyr.attention.q_lin = prune_layer(lyr.attention.q_lin, index, dim=1)\\n'\n"
               "        '                    lyr.attention.k_lin = prune_layer(lyr.attention.k_lin, index, dim=1)\\n'\n"
               "        '                if lyr.attention.v_lin is not None:\\n'\n"
               "        '                    lyr.attention.v_lin = prune_layer(lyr.attention.v_lin, index, dim=1)\\n'\n"
               "        '                    lyr.attention.out_lin = prune_layer(lyr.attention.out_lin, index, "
               "dim=0)\\n'\n"
               "        '                    prune_layer_norm(lyr.attn_output.LayerNorm, index)\\n'\n"
               "        '                if lyr.ffn_lin1 is not None:\\n'\n"
               "        '                    lyr.ffn_lin1 = prune_layer(lyr.ffn_lin1, index, dim=1)\\n'\n"
               "        '                    lyr.ffn_lin2 = prune_layer(lyr.ffn_lin2, index, dim=0)\\n'\n"
               "        '                    prune_layer_norm(lyr.ffn_output.LayerNorm, index)',\n"
               "        'cofi_utils: hidden_z layer loop — bert/distilbert')\n"
               '\n'
               '    # print loop — model-agnostic\n'
               '    patch_file(utils_path,\n'
               "        '    for layer in range(0, 12):\\n'\n"
               '        \'        print("Layer:", layer)\\n\'\n'
               "        '        if bert.encoder.layer[layer].attention.self.query is not None:\\n'\n"
               '        \'            print("query:", '
               "bert.encoder.layer[layer].attention.self.query.weight.shape)\\n'\n"
               '        \'            print("key:", bert.encoder.layer[layer].attention.self.key.weight.shape)\\n\'\n'
               "        '        else:\\n'\n"
               '        \'            print("query:", None)\\n\'\n'
               '        \'            print("key:", None)\\n\'\n'
               "        '        if bert.encoder.layer[layer].attention.self.value is not None:\\n'\n"
               '        \'            print("value:", '
               "bert.encoder.layer[layer].attention.self.value.weight.shape)\\n'\n"
               '        \'            print("output:", '
               "bert.encoder.layer[layer].attention.output.dense.weight.shape)\\n'\n"
               "        '        else:\\n'\n"
               '        \'            print("value:", None)\\n\'\n'
               '        \'            print("output:", None)\\n\'\n'
               "        '        if bert.encoder.layer[layer].intermediate.dense is not None:\\n'\n"
               '        \'            print("up:", bert.encoder.layer[layer].intermediate.dense.weight.shape)\\n\'\n'
               '        \'            print("down:", bert.encoder.layer[layer].output.dense.weight.shape)\\n\'\n'
               "        '        else:\\n'\n"
               '        \'            print("up", None)\\n\'\n'
               '        \'            print("down", None)\',\n'
               "        '    layers = _get_layers(bert)\\n'\n"
               "        '    for layer in range(num_layers):\\n'\n"
               '        \'        print("Layer:", layer)\\n\'\n'
               "        '        lyr = layers[layer]\\n'\n"
               '        \'        if hasattr(lyr, "attention") and hasattr(lyr.attention, "self"):\\n\'\n'
               "        '            q = lyr.attention.self.query\\n'\n"
               "        '            v = lyr.attention.self.value\\n'\n"
               "        '            up = lyr.intermediate.dense\\n'\n"
               "        '            down = lyr.output.dense\\n'\n"
               "        '        else:\\n'\n"
               '        \'            q = getattr(lyr.attention, "q_lin", None)\\n\'\n'
               '        \'            v = getattr(lyr.attention, "v_lin", None)\\n\'\n'
               '        \'            up = getattr(lyr, "ffn_lin1", None)\\n\'\n'
               '        \'            down = getattr(lyr, "ffn_lin2", None)\\n\'\n'
               '        \'        print("query:", q.weight.shape if q is not None else None)\\n\'\n'
               '        \'        print("key:", q.weight.shape if q is not None else None)\\n\'\n'
               '        \'        print("value:", v.weight.shape if v is not None else None)\\n\'\n'
               '        \'        print("up:", up.weight.shape if up is not None else None)\\n\'\n'
               '        \'        print("down:", down.weight.shape if down is not None else None)\',\n'
               "        'cofi_utils: print loop — model-agnostic')\n"
               '\n'
               '    # pre_classifier (distilbert) + classifier + safe pooler — BOTH dims for square matrices\n'
               '    patch_file(utils_path,\n'
               '        \'        if hasattr(model, "classifier"):\\n\'\n'
               '        \'            if hasattr(model.classifier, "dense"):\\n\'\n'
               "        '                model.classifier.dense = prune_linear_layer(model.classifier.dense, index, "
               "dim=1)\\n'\n"
               '        \'        if hasattr(model, "cls"):\\n\'\n'
               '        \'            if hasattr(model.cls, "dense"):\\n\'\n'
               "        '                model.cls.dense = prune_linear_layer(model.classifier.dense, index, "
               "dim=1)\\n'\n"
               '        \'        if hasattr(bert.pooler, "dense"):\\n\'\n'
               "        '            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)\\n'\n"
               '        \'        if hasattr(model, "qa_outputs"):\\n\'\n'
               "        '            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)',\n"
               '        \'        if hasattr(model, "pre_classifier"):\\n\'\n'
               "        '            model.pre_classifier = prune_linear_layer(model.pre_classifier, index, "
               "dim=1)\\n'\n"
               "        '            model.pre_classifier = prune_linear_layer(model.pre_classifier, index, "
               "dim=0)\\n'\n"
               '        \'        if hasattr(model, "classifier"):\\n\'\n'
               '        \'            if hasattr(model.classifier, "dense"):\\n\'\n'
               "        '                model.classifier.dense = prune_linear_layer(model.classifier.dense, index, "
               "dim=1)\\n'\n"
               "        '            elif isinstance(model.classifier, torch.nn.Linear):\\n'\n"
               "        '                model.classifier = prune_linear_layer(model.classifier, index, dim=1)\\n'\n"
               '        \'        if hasattr(model, "cls"):\\n\'\n'
               '        \'            if hasattr(model.cls, "dense"):\\n\'\n'
               "        '                model.cls.dense = prune_linear_layer(model.classifier.dense, index, "
               "dim=1)\\n'\n"
               '        \'        if hasattr(bert, "pooler") and bert.pooler is not None and hasattr(bert.pooler, '
               '"dense"):\\n\'\n'
               "        '            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)\\n'\n"
               "        '            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=0)\\n'\n"
               '        \'        if hasattr(model, "qa_outputs"):\\n\'\n'
               "        '            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)',\n"
               "        'cofi_utils: pre_classifier + classifier + pooler — both dims, safe None check')\n"
               '\n'
               '    # hidden_z embeddings — guard token_type_embeddings (distilbert has none)\n'
               '    patch_file(utils_path,\n'
               "        '        bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
               "        '            bert.embeddings.token_type_embeddings.weight.index_select(1, "
               "index).clone().detach())\\n'\n"
               "        '        bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]',\n"
               '        \'        if hasattr(bert.embeddings, "token_type_embeddings"):\\n\'\n'
               "        '            bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
               "        '                bert.embeddings.token_type_embeddings.weight.index_select(1, "
               "index).clone().detach())\\n'\n"
               "        '            bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]',\n"
               "        'cofi_utils: guard token_type_embeddings in prune_model_with_z hidden_z')\n"
               '\n'
               '    # prune_intermediate_layers — model-agnostic\n'
               '    patch_file(utils_path,\n'
               "        'def prune_intermediate_layers(model, keep_dims):\\n'\n"
               '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\\n\'\n'
               "        '    device = model.device\\n'\n"
               "        '    for layer in keep_dims:\\n'\n"
               "        '        if len(keep_dims[layer]) == 0:\\n'\n"
               "        '            bert.encoder.layer[layer].intermediate.dense = None\\n'\n"
               "        '            bert.encoder.layer[layer].output.dense = None\\n'\n"
               "        '        else:\\n'\n"
               "        '            bert.encoder.layer[layer].intermediate.dense = "
               'prune_linear_layer(bert.encoder.layer[layer].intermediate.dense, '
               "index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)\\n'\n"
               "        '            bert.encoder.layer[layer].output.dense = "
               'prune_linear_layer(bert.encoder.layer[layer].output.dense, '
               "index=torch.LongTensor(keep_dims[layer]).to(device), dim=1)',\n"
               "        'def prune_intermediate_layers(model, keep_dims):\\n'\n"
               '        \'    if hasattr(model, "bert"):\\n\'\n'
               "        '        bert = model.bert\\n'\n"
               '        \'    elif hasattr(model, "roberta"):\\n\'\n'
               "        '        bert = model.roberta\\n'\n"
               '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
               "        '        bert = model.distilbert\\n'\n"
               "        '    else:\\n'\n"
               '        \'        raise ValueError(f"Unknown model type: {type(model)}")\\n\'\n'
               "        '    device = model.device\\n'\n"
               "        '    layers = _get_layers(bert)\\n'\n"
               "        '    for layer in keep_dims:\\n'\n"
               "        '        lyr = layers[layer]\\n'\n"
               '        \'        is_distil = hasattr(lyr, "ffn_lin1")\\n\'\n'
               "        '        if len(keep_dims[layer]) == 0:\\n'\n"
               "        '            if is_distil:\\n'\n"
               "        '                lyr.ffn_lin1 = None\\n'\n"
               "        '                lyr.ffn_lin2 = None\\n'\n"
               "        '            else:\\n'\n"
               "        '                lyr.intermediate.dense = None\\n'\n"
               "        '                lyr.output.dense = None\\n'\n"
               "        '        else:\\n'\n"
               "        '            idx = torch.LongTensor(keep_dims[layer]).to(device)\\n'\n"
               "        '            if is_distil:\\n'\n"
               "        '                lyr.ffn_lin1 = prune_linear_layer(lyr.ffn_lin1, index=idx, dim=0)\\n'\n"
               "        '                lyr.ffn_lin2 = prune_linear_layer(lyr.ffn_lin2, index=idx, dim=1)\\n'\n"
               "        '            else:\\n'\n"
               "        '                lyr.intermediate.dense = prune_linear_layer(lyr.intermediate.dense, "
               "index=idx, dim=0)\\n'\n"
               "        '                lyr.output.dense = prune_linear_layer(lyr.output.dense, index=idx, dim=1)',\n"
               "        'cofi_utils: prune_intermediate_layers — model-agnostic')\n"
               '\n'
               '    # update_params — full model-agnostic rewrite (this is a SEPARATE function\n'
               '    # from prune_model_with_z, runs BEFORE it, had its own hardcoded bert/roberta-only check)\n'
               '    patch_file(utils_path,\n'
               "        'def update_params(model, zs):\\n'\n"
               '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\\n\'\n'
               "        '\\n'\n"
               "        '    config = model.config\\n'\n"
               "        '    hidden_dims = config.hidden_size\\n'\n"
               "        '    num_heads = config.num_attention_heads\\n'\n"
               "        '    dims_per_head = hidden_dims // num_heads\\n'\n"
               "        '    num_layers = config.num_hidden_layers',\n"
               "        'def update_params(model, zs):\\n'\n"
               '        \'    if hasattr(model, "bert"):\\n\'\n'
               "        '        bert = model.bert\\n'\n"
               "        '        num_layers = model.config.num_hidden_layers\\n'\n"
               "        '        is_distil = False\\n'\n"
               '        \'    elif hasattr(model, "roberta"):\\n\'\n'
               "        '        bert = model.roberta\\n'\n"
               "        '        num_layers = model.config.num_hidden_layers\\n'\n"
               "        '        is_distil = False\\n'\n"
               '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
               "        '        bert = model.distilbert\\n'\n"
               "        '        num_layers = model.config.n_layers\\n'\n"
               "        '        is_distil = True\\n'\n"
               "        '    else:\\n'\n"
               '        \'        raise ValueError(f"Unknown model type: {type(model)}")\\n\'\n'
               "        '\\n'\n"
               "        '    config = model.config\\n'\n"
               "        '    if is_distil:\\n'\n"
               "        '        hidden_dims = config.dim\\n'\n"
               "        '        num_heads = config.n_heads\\n'\n"
               "        '    else:\\n'\n"
               "        '        hidden_dims = config.hidden_size\\n'\n"
               "        '        num_heads = config.num_attention_heads\\n'\n"
               "        '    dims_per_head = hidden_dims // num_heads',\n"
               "        'cofi_utils: update_params — full model-agnostic rewrite')\n"
               '\n'
               '    # update_params: intermediate_z loop\n'
               '    patch_file(utils_path,\n'
               '        \'        if "intermediate_z" in zs:\\n\'\n'
               "        '            for layer in range(num_layers):\\n'\n"
               '        \'                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()\\n\'\n'
               "        '                bert.encoder.layer[layer].output.dense.weight.data = "
               "bert.encoder.layer[layer].output.dense.weight.data.mul(intermediate_z)\\n'\n"
               '        \'                if "mlp_z" in zs:\\n\'\n'
               '        \'                    mlp_z = zs["mlp_z"][layer].cpu()\\n\'\n'
               "        '                    bert.encoder.layer[layer].output.dense.weight.data = "
               "bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)\\n'\n"
               "        '                    bert.encoder.layer[layer].output.dense.bias.data = "
               "bert.encoder.layer[layer].output.dense.bias.data.mul(mlp_z)',\n"
               '        \'        if "intermediate_z" in zs:\\n\'\n'
               "        '            layers = _get_layers(bert)\\n'\n"
               "        '            for layer in range(num_layers):\\n'\n"
               '        \'                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()\\n\'\n'
               "        '                lyr = layers[layer]\\n'\n"
               "        '                down = lyr.ffn_lin2 if is_distil else lyr.output.dense\\n'\n"
               "        '                down.weight.data = down.weight.data.mul(intermediate_z)\\n'\n"
               '        \'                if "mlp_z" in zs:\\n\'\n'
               '        \'                    mlp_z = zs["mlp_z"][layer].cpu()\\n\'\n'
               "        '                    down.weight.data = down.weight.data.transpose(0, "
               "1).mul(mlp_z).transpose(0, 1)\\n'\n"
               "        '                    down.bias.data = down.bias.data.mul(mlp_z)',\n"
               "        'cofi_utils: update_params intermediate_z loop — model-agnostic')\n"
               '\n'
               '    # update_params: head_z loop\n'
               '    patch_file(utils_path,\n'
               '        \'        if "head_z" in zs:\\n\'\n'
               "        '            for layer in range(num_layers):\\n'\n"
               '        \'                head_z = zs["head_z"][layer].cpu().squeeze().clone()\\n\'\n'
               "        '                head_z = torch.repeat_interleave(head_z, dims_per_head)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.value.weight.data = "
               'bert.encoder.layer[layer].attention.self.value.weight.transpose(0, 1).data.mul(head_z).transpose(0, '
               "1)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.value.bias.data = "
               "bert.encoder.layer[layer].attention.self.value.bias.data.mul(head_z)\\n'\n"
               '        \'                if "head_layer_z" in zs:\\n\'\n'
               '        \'                    head_layer_z = zs["head_layer_z"][layer].cpu()\\n\'\n'
               "        '                    bert.encoder.layer[layer].attention.output.dense.weight.data = "
               "bert.encoder.layer[\\n'\n"
               "        '                        layer].attention.output.dense.weight.transpose(0, "
               "1).data.mul(head_layer_z).transpose(0, 1)\\n'\n"
               "        '                    bert.encoder.layer[layer].attention.output.dense.bias.data = "
               "bert.encoder.layer[\\n'\n"
               "        '                        layer].attention.output.dense.bias.data.mul(head_layer_z)',\n"
               '        \'        if "head_z" in zs:\\n\'\n'
               "        '            layers = _get_layers(bert)\\n'\n"
               "        '            for layer in range(num_layers):\\n'\n"
               '        \'                head_z = zs["head_z"][layer].cpu().squeeze().clone()\\n\'\n'
               "        '                head_z = torch.repeat_interleave(head_z, dims_per_head)\\n'\n"
               "        '                lyr = layers[layer]\\n'\n"
               "        '                if is_distil:\\n'\n"
               "        '                    v, o = lyr.attention.v_lin, lyr.attention.out_lin\\n'\n"
               "        '                else:\\n'\n"
               "        '                    v, o = lyr.attention.self.value, lyr.attention.output.dense\\n'\n"
               "        '                v.weight.data = v.weight.transpose(0, 1).data.mul(head_z).transpose(0, "
               "1)\\n'\n"
               "        '                v.bias.data = v.bias.data.mul(head_z)\\n'\n"
               '        \'                if "head_layer_z" in zs:\\n\'\n'
               '        \'                    head_layer_z = zs["head_layer_z"][layer].cpu()\\n\'\n'
               "        '                    o.weight.data = o.weight.transpose(0, "
               "1).data.mul(head_layer_z).transpose(0, 1)\\n'\n"
               "        '                    o.bias.data = o.bias.data.mul(head_layer_z)',\n"
               "        'cofi_utils: update_params head_z loop — model-agnostic')\n"
               '\n'
               '    # update_params: hidden_z loop\n'
               '    patch_file(utils_path,\n'
               '        \'        if "hidden_z" in zs:\\n\'\n'
               '        \'            hidden_z = zs["hidden_z"].cpu().squeeze().clone()\\n\'\n'
               "        '            bert.embeddings.word_embeddings.weight.data =\\\\\\n'\n"
               "        '                bert.embeddings.word_embeddings.weight.data.mul(hidden_z)\\n'\n"
               "        '            bert.embeddings.position_embeddings.weight.data = \\\\\\n'\n"
               "        '                bert.embeddings.position_embeddings.weight.data.mul(hidden_z)\\n'\n"
               "        '            bert.embeddings.token_type_embeddings.weight.data = \\\\\\n'\n"
               "        '                bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)\\n'\n"
               "        '            for layer in range(num_layers):\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.key.weight.data = "
               "bert.encoder.layer[layer].attention.self.key.weight.data.mul(hidden_z)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.query.weight.data = "
               "bert.encoder.layer[layer].attention.self.query.weight.data.mul(hidden_z)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.value.weight.data = "
               "bert.encoder.layer[layer].attention.self.value.weight.data.mul(hidden_z)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.output.dense.weight.data = "
               'bert.encoder.layer[layer].attention.output.dense.weight.data.transpose(0, '
               "1).mul(hidden_z).transpose(0, 1)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.output.dense.bias.data = "
               "bert.encoder.layer[layer].attention.output.dense.bias.data.mul(hidden_z)\\n'\n"
               "        '                bert.encoder.layer[layer].intermediate.dense.weight.data = "
               "bert.encoder.layer[layer].intermediate.dense.weight.data.mul(hidden_z)\\n'\n"
               "        '                bert.encoder.layer[layer].output.dense.weight.data = "
               "bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)\\n'\n"
               '        \'            if hasattr(bert.pooler, "dense"):\\n\'\n'
               "        '                bert.pooler.dense.weight.data = "
               "bert.pooler.dense.weight.data.mul(hidden_z)\\n'\n"
               '        \'            if hasattr(model, "qa_outputs"):\\n\'\n'
               "        '                model.qa_outputs.weight.data = model.qa_outputs.weight.data.mul(hidden_z)',\n"
               '        \'        if "hidden_z" in zs:\\n\'\n'
               '        \'            hidden_z = zs["hidden_z"].cpu().squeeze().clone()\\n\'\n'
               "        '            bert.embeddings.word_embeddings.weight.data = \\\\\\n'\n"
               "        '                bert.embeddings.word_embeddings.weight.data.mul(hidden_z)\\n'\n"
               "        '            bert.embeddings.position_embeddings.weight.data = \\\\\\n'\n"
               "        '                bert.embeddings.position_embeddings.weight.data.mul(hidden_z)\\n'\n"
               '        \'            if hasattr(bert.embeddings, "token_type_embeddings"):\\n\'\n'
               "        '                bert.embeddings.token_type_embeddings.weight.data = \\\\\\n'\n"
               "        '                    bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)\\n'\n"
               "        '            layers = _get_layers(bert)\\n'\n"
               "        '            for layer in range(num_layers):\\n'\n"
               "        '                lyr = layers[layer]\\n'\n"
               "        '                if is_distil:\\n'\n"
               "        '                    q, k, v, o = lyr.attention.q_lin, lyr.attention.k_lin, "
               "lyr.attention.v_lin, lyr.attention.out_lin\\n'\n"
               "        '                    up, down = lyr.ffn_lin1, lyr.ffn_lin2\\n'\n"
               "        '                else:\\n'\n"
               "        '                    q, k, v = lyr.attention.self.query, lyr.attention.self.key, "
               "lyr.attention.self.value\\n'\n"
               "        '                    o = lyr.attention.output.dense\\n'\n"
               "        '                    up, down = lyr.intermediate.dense, lyr.output.dense\\n'\n"
               "        '                k.weight.data = k.weight.data.mul(hidden_z)\\n'\n"
               "        '                q.weight.data = q.weight.data.mul(hidden_z)\\n'\n"
               "        '                v.weight.data = v.weight.data.mul(hidden_z)\\n'\n"
               "        '                o.weight.data = o.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, "
               "1)\\n'\n"
               "        '                o.bias.data = o.bias.data.mul(hidden_z)\\n'\n"
               "        '                up.weight.data = up.weight.data.mul(hidden_z)\\n'\n"
               "        '                down.weight.data = down.weight.data.transpose(0, "
               "1).mul(hidden_z).transpose(0, 1)\\n'\n"
               '        \'            if hasattr(model, "pre_classifier"):\\n\'\n'
               "        '                model.pre_classifier.weight.data = "
               "model.pre_classifier.weight.data.mul(hidden_z)\\n'\n"
               '        \'            elif hasattr(bert, "pooler") and bert.pooler is not None and '
               'hasattr(bert.pooler, "dense"):\\n\'\n'
               "        '                bert.pooler.dense.weight.data = "
               "bert.pooler.dense.weight.data.mul(hidden_z)\\n'\n"
               '        \'            if hasattr(model, "qa_outputs"):\\n\'\n'
               "        '                model.qa_outputs.weight.data = model.qa_outputs.weight.data.mul(hidden_z)',\n"
               "        'cofi_utils: update_params hidden_z loop — model-agnostic')\n"
               '\n'
               '    # ── models/l0_module.py: config attr name normalization ──────────────────\n'
               "    print('\\n  Patching models/l0_module.py...')\n"
               "    l0_path = os.path.join(DISTIL_REPO_DIR, 'models', 'l0_module.py')\n"
               '    patch_file(l0_path,\n'
               "        '        self.hidden_size = config.hidden_size\\n'\n"
               "        '        self.intermediate_size = config.intermediate_size \\n'\n"
               "        '        self.num_attention_heads = config.num_attention_heads\\n'\n"
               "        '        self.mlp_num_per_layer = 1\\n'\n"
               "        '        self.dim_per_head = self.hidden_size // self.num_attention_heads \\n'\n"
               "        '        self.num_hidden_layers = config.num_hidden_layers\\n'\n"
               "        '        self.vocab_size = config.vocab_size',\n"
               '        \'        self.hidden_size = getattr(config, "hidden_size", getattr(config, "dim", '
               "None))\\n'\n"
               '        \'        self.intermediate_size = getattr(config, "intermediate_size", getattr(config, '
               '"hidden_dim", None))\\n\'\n'
               '        \'        self.num_attention_heads = getattr(config, "num_attention_heads", getattr(config, '
               '"n_heads", None))\\n\'\n'
               "        '        self.mlp_num_per_layer = 1\\n'\n"
               "        '        self.dim_per_head = self.hidden_size // self.num_attention_heads\\n'\n"
               '        \'        self.num_hidden_layers = getattr(config, "num_hidden_layers", getattr(config, '
               '"n_layers", None))\\n\'\n'
               "        '        self.vocab_size = config.vocab_size',\n"
               "        'l0_module: config getattr fix for distilbert (note: file has trailing'\n"
               "        ' whitespace on some original lines — watch for [not found] here)')\n"
               '\n'
               '    # ── trainer/trainer.py ─────────────────────────────────────────────────────\n'
               "    print('\\n  Patching trainer/trainer.py...')\n"
               "    trainer_path = os.path.join(DISTIL_REPO_DIR, 'trainer', 'trainer.py')\n"
               '    patch_file(trainer_path,\n'
               "        '* (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)',\n"
               "        '',\n"
               "        'trainer: remove world_size')\n"
               '    patch_file(trainer_path,\n'
               "        '                if self.start_prune:\\n                    zs = "
               "self.l0_module.forward(training=True)',\n"
               "        '                if self.start_prune and self.l0_module is not None:\\n                    zs "
               "= self.l0_module.forward(training=True)',\n"
               "        'trainer: guard l0_module None')\n"
               '    patch_file(trainer_path,\n'
               '        "torch.save(self.l0_module, os.path.join(output_dir, \'l0_module.pt\'))",\n'
               '        "if self.l0_module is not None:\\n            torch.save(self.l0_module, '
               'os.path.join(output_dir, \'l0_module.pt\'))",\n'
               "        'trainer: guard l0_module save')\n"
               '\n'
               '    # Dynamic teacher layer indices — DistilBERT teacher has only 6 hidden-state\n'
               '    # layers, not 12, so the hardcoded [2,5,8,11] indices would IndexError.\n'
               '    patch_file(trainer_path,\n'
               "        '                else:\\n                    specified_teacher_layers = [2, 5, 8, 11]',\n"
               "        '                else:\\n'\n"
               "        '                    n_teacher_layers = teacher_outputs[2].__len__() - 1\\n'\n"
               "        '                    if n_teacher_layers >= 12:\\n'\n"
               "        '                        specified_teacher_layers = [2, 5, 8, 11]\\n'\n"
               "        '                    else:\\n'\n"
               "        '                        step = max(1, n_teacher_layers // 4)\\n'\n"
               "        '                        specified_teacher_layers = [min(i * step, n_teacher_layers - 1) for i "
               "in range(1, 5)]\\n'\n"
               "        '                        specified_teacher_layers = sorted(set(specified_teacher_layers))\\n'\n"
               "        '                        while len(specified_teacher_layers) < 4:\\n'\n"
               "        '                            specified_teacher_layers.append(n_teacher_layers - 1)',\n"
               "        'trainer: dynamic teacher layer indices for <12-layer teacher')\n"
               '\n'
               "    print('\\nAll patches done.')\n"
               "    print('NOTE: modeling_distilbert.py was written as-is (already fully patched —')\n"
               "    print('      includes del self.ffn shared-tensor fix, zero-head guards, etc.)')\n"
               "    print('Re-run block 0 to verify all lines show [already patched].')\n"
               '\n'
               '\n'
               '# ── Block 1: Download pre-tuned models ────────────────────────────────────────\n'
               'def block1(tasks):\n'
               "    header('BLOCK 1 — Download Pre-tuned DistilBERT Models')\n"
               '    from transformers import AutoModelForSequenceClassification, AutoTokenizer\n'
               '\n'
               '    for task in tasks:\n'
               "        hf_id = DISTIL_PRETRAINED_FT.get(task, '')\n"
               '        if not hf_id:\n'
               "            print(f'[SKIP] {task}: no HuggingFace model ID set in DISTIL_PRETRAINED_FT')\n"
               '            continue\n'
               '        out = ft_dir(task)\n'
               '        os.makedirs(out, exist_ok=True)\n'
               '        if model_saved(out):\n'
               "            print(f'[SKIP] {task}: already at {out}')\n"
               '            continue\n'
               "        print(f'Downloading {hf_id} -> {out} ...')\n"
               '        model = AutoModelForSequenceClassification.from_pretrained(\n'
               '            hf_id, trust_remote_code=True)\n'
               '        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)\n'
               '        model.save_pretrained(out)\n'
               '        tok.save_pretrained(out)\n'
               '        del model, tok\n'
               "        print(f'  Saved.')\n"
               '\n'
               "    print('\\nBlock 1 done.')\n"
               '\n'
               '\n'
               '# ── Block 2: CoFi pruning ──────────────────────────────────────────────────────\n'
               'def block2(tasks):\n'
               "    header('BLOCK 2 — CoFi Pruning (DistilBERT)')\n"
               '\n'
               '    if DISTIL_REPO_DIR not in sys.path:\n'
               '        sys.path.insert(0, DISTIL_REPO_DIR)\n'
               '\n'
               "    env = {**os.environ, 'HF_DATASETS_TRUST_REMOTE_CODE': '1'}\n"
               '\n'
               '    for task in tasks:\n'
               '        ft  = ft_dir(task)\n'
               '        out = pr_dir(task)\n'
               '        cfg = task_cfg(task)\n'
               '        os.makedirs(out, exist_ok=True)\n'
               '\n'
               '        if not model_saved(ft):\n'
               "            print(f'[ERROR] {task}: fine-tuned model missing at {ft}. Run --block 1 first.')\n"
               '            continue\n'
               '\n'
               "        best = os.path.join(out, 'best')\n"
               '        if model_saved(best):\n'
               "            print(f'[SKIP] {task}: already pruned at {best}')\n"
               '            continue\n'
               '\n'
               "        print(f'\\nPruning distilbert/{task} -> {out}')\n"
               "        log_file = os.path.join(out, 'pruning_log.txt')\n"
               "        print(f'Log:     {log_file}')\n"
               "        print(f'Monitor: tail -f {log_file}')\n"
               "        print('Ctrl+C stops safely — resumes from last checkpoint on next run.\\n')\n"
               '\n'
               '        cmd = [\n'
               '            sys.executable,\n'
               "            os.path.join(DISTIL_REPO_DIR, 'run_glue_prune.py'),\n"
               "            '--model_name_or_path', 'distilbert-base-uncased',\n"
               "            '--task_name', task,\n"
               "            '--do_train', '--do_eval',\n"
               "            '--max_seq_length', '128',\n"
               "            '--per_device_train_batch_size', '32',\n"
               "            '--per_device_eval_batch_size', '32',\n"
               "            '--learning_rate', '2e-5',\n"
               "            '--reg_learning_rate', '0.01',\n"
               "            '--num_train_epochs', str(cfg['prune_epochs']),\n"
               "            '--output_dir', out,\n"
               "            '--save_steps', str(cfg['save_steps']),\n"
               "            '--save_total_limit', '2',\n"
               "            '--eval_steps', str(cfg['eval_steps']),\n"
               "            '--eval_strategy', 'steps',\n"
               "            '--seed', str(SEED),\n"
               "            '--pruning_type', 'structured_heads+structured_mlp+hidden+layer',\n"
               "            '--target_sparsity', str(SPARSITY),\n"
               "            '--sparsity_epsilon', '0.01',\n"
               "            '--freeze_embeddings',\n"
               "            '--do_distill', '--do_layer_distill',\n"
               "            '--distillation_path', ft,\n"
               "            '--distill_ce_loss_alpha', '0.1',\n"
               "            '--distill_loss_alpha', '0.9',\n"
               "            '--distill_temp', '2',\n"
               "            '--layer_distill_version', str(cfg['layer_distill_v']),\n"
               "            '--prepruning_finetune_epochs', str(cfg['prepruning']),\n"
               "            '--lagrangian_warmup_epochs', str(cfg['lag_warmup']),\n"
               "            '--scheduler_type', 'linear',\n"
               "            '--local_rank', '-1',\n"
               "            '--report_to', 'none',\n"
               '        ]\n'
               '\n'
               "        with open(log_file, 'w') as log:\n"
               '            proc = subprocess.Popen(\n'
               '                cmd,\n'
               '                stdout=subprocess.PIPE,\n'
               '                stderr=subprocess.STDOUT,\n'
               '                text=True,\n'
               '                cwd=DISTIL_REPO_DIR,\n'
               '                env=env,\n'
               '            )\n'
               '            for line in proc.stdout:\n'
               '                sys.stdout.write(line)\n'
               '                sys.stdout.flush()\n'
               '                log.write(line)\n'
               '                log.flush()\n'
               '            proc.wait()\n'
               '\n'
               '        if model_saved(best):\n'
               "            print(f'\\n[DONE] {task}: best model at {best}')\n"
               '        else:\n'
               "            print(f'\\n[WARNING] {task}: no best/ checkpoint. Check {log_file}')\n"
               '\n'
               "    print('\\nBlock 2 done.')\n"
               '\n'
               '\n'
               '# ── Block 3: Evaluation ────────────────────────────────────────────────────────\n'
               'def block3(tasks):\n'
               "    header('BLOCK 3 — Evaluation (DistilBERT)')\n"
               '\n'
               '    import torch\n'
               '    from torch.utils.data import DataLoader\n'
               '    from transformers import AutoModelForSequenceClassification, AutoTokenizer\n'
               '    from datasets import load_dataset\n'
               '    import evaluate as hf_evaluate\n'
               '\n'
               '    if DISTIL_REPO_DIR not in sys.path:\n'
               '        sys.path.insert(0, DISTIL_REPO_DIR)\n'
               '\n'
               '    def run_evaluation(model_path, task, label, out_dir):\n'
               '        result_file = eval_path(out_dir)\n'
               '        if os.path.exists(result_file):\n'
               "            print(f'[SKIP] {label}/{task}: already evaluated.')\n"
               '            r = json.load(open(result_file))\n'
               '            for k, v in r.items():\n'
               "                print(f'  {k}: {v}')\n"
               '            return r\n'
               '\n'
               "        print(f'\\nEvaluating [{label}] on {task} ...')\n"
               "        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
               "        print(f'  Device: {device}')\n"
               '\n'
               '        tok_path = (ft_dir(task)\n'
               '                    if not os.path.exists(\n'
               "                        os.path.join(model_path, 'tokenizer_config.json'))\n"
               '                    else model_path)\n'
               '        tok = AutoTokenizer.from_pretrained(tok_path)\n'
               '\n'
               "        if label.lower().strip().startswith('distilbert pruned'):\n"
               '            from models.modeling_distilbert import CoFiDistilBertForSequenceClassification\n'
               '            from utils.cofi_utils import load_zs, load_model\n'
               '            zs = load_zs(model_path)\n'
               '            if zs is None:\n'
               '                model = CoFiDistilBertForSequenceClassification.from_pretrained(model_path)\n'
               '            else:\n'
               '                model = load_model(model_path, CoFiDistilBertForSequenceClassification, zs)\n'
               '        else:\n'
               '            model = AutoModelForSequenceClassification.from_pretrained(model_path)\n'
               '\n'
               '        n_params = sum(p.numel() for p in model.parameters())\n'
               '        mem_mb   = n_params * 4 / 1e6\n'
               '        model    = model.to(device).eval()\n'
               '\n'
               "        if task == 'mnli':\n"
               "            ds = load_dataset('glue', 'mnli',\n"
               "                              trust_remote_code=True)['validation_matched']\n"
               '        else:\n'
               "            ds = load_dataset('glue', task,\n"
               "                              trust_remote_code=True)['validation']\n"
               '\n'
               '        col_map = {\n'
               "            'sst2': ('sentence',  None),\n"
               "            'qnli': ('question',  'sentence'),\n"
               "            'mnli': ('premise',   'hypothesis'),\n"
               "            'qqp' : ('question1', 'question2'),\n"
               "            'rte' : ('sentence1', 'sentence2'),\n"
               '        }\n'
               '        col_a, col_b = col_map[task]\n'
               '        ds = ds.select(range(min(1000, len(ds))))\n'
               '\n'
               '        def tokenize(batch):\n'
               '            args = ((batch[col_a],) if col_b is None\n'
               '                    else (batch[col_a], batch[col_b]))\n'
               "            return tok(*args, padding='max_length', truncation=True,\n"
               '                       max_length=128, return_tensors=None)\n'
               '\n'
               '        ds = ds.map(tokenize, batched=True,\n'
               '                    remove_columns=[c for c in ds.column_names\n'
               "                                    if c not in ['label', 'labels', 'idx']])\n"
               "        ds.set_format('torch')\n"
               '        loader = DataLoader(ds, batch_size=32)\n'
               '\n'
               '        all_preds, all_labels = [], []\n'
               '        total_time, total_examples = 0.0, 0\n'
               '\n'
               '        # warmup\n'
               '        with torch.no_grad():\n'
               '            for batch in loader:\n'
               '                inp = {k: v.to(device) for k, v in batch.items()\n'
               "                       if k in ['input_ids', 'attention_mask']}\n"
               '                model(**inp)\n'
               '                break\n'
               '\n'
               '        with torch.no_grad():\n'
               '            for batch in loader:\n'
               "                labels = batch.pop('label', batch.pop('labels', None))\n"
               '                inp = {k: v.to(device) for k, v in batch.items()\n'
               "                       if k in ['input_ids', 'attention_mask']}\n"
               '                if torch.cuda.is_available():\n'
               '                    torch.cuda.synchronize()\n'
               '                t0 = time.perf_counter()\n'
               '                out = model(**inp)\n'
               '                if torch.cuda.is_available():\n'
               '                    torch.cuda.synchronize()\n'
               '                t1 = time.perf_counter()\n'
               '                total_time     += (t1 - t0)\n'
               '                total_examples += out.logits.shape[0]\n'
               '                all_preds.extend(out.logits.argmax(-1).cpu().numpy().tolist())\n'
               '                if labels is not None:\n'
               '                    all_labels.extend(labels.cpu().numpy().tolist())\n'
               '\n'
               '        latency_ms = (total_time / total_examples) * 1000\n'
               '        throughput  = total_examples / total_time\n'
               '\n'
               "        if task == 'qqp':\n"
               "            score = hf_evaluate.load('glue', 'qqp').compute(\n"
               "                predictions=all_preds, references=all_labels)['f1']\n"
               "            metric_name = 'F1'\n"
               "        elif task == 'mnli':\n"
               "            score = hf_evaluate.load('glue', 'mnli').compute(\n"
               "                predictions=all_preds, references=all_labels)['accuracy']\n"
               "            metric_name = 'Accuracy'\n"
               '        else:\n'
               "            score = hf_evaluate.load('glue', task).compute(\n"
               "                predictions=all_preds, references=all_labels)['accuracy']\n"
               "            metric_name = 'Accuracy'\n"
               '\n'
               '        # Sparsity: exclude embeddings\n'
               '        n_non_emb   = n_params - DISTILBERT_EMBEDDING_PARAMS\n'
               '        sparsity_pct = max(0.0, (1 - n_non_emb / DISTILBERT_BASE_PARAMS) * 100)\n'
               '\n'
               '        results = {\n'
               "            'label':          label,\n"
               "            'task':           task,\n"
               "            'n_params':       n_params,\n"
               "            'memory_mb':      round(mem_mb, 2),\n"
               "            'latency_ms':     round(latency_ms, 4),\n"
               "            'throughput_eps': round(throughput, 2),\n"
               "            'sparsity_pct':   round(sparsity_pct, 2),\n"
               '            metric_name:      round(score, 4),\n'
               '        }\n'
               '\n'
               '        W = 50\n'
               "        print('=' * W)\n"
               "        print(f'  {label} — {task.upper()}')\n"
               "        print('=' * W)\n"
               "        print(f'  {metric_name:<22}: {score:.4f}')\n"
               "        print(f'  Memory (MB)          : {mem_mb:.1f}')\n"
               "        print(f'  Latency (ms/example) : {latency_ms:.3f}')\n"
               "        print(f'  Throughput (ex/sec)  : {throughput:.1f}')\n"
               "        print(f'  Sparsity %           : {sparsity_pct:.1f}%')\n"
               "        print(f'  Parameters           : {n_params:,}')\n"
               "        print('=' * W)\n"
               '\n'
               '        os.makedirs(out_dir, exist_ok=True)\n'
               "        with open(result_file, 'w') as f:\n"
               '            json.dump(results, f, indent=2)\n'
               "        print(f'  Saved to {result_file}')\n"
               '\n'
               '        del model\n'
               '        if torch.cuda.is_available():\n'
               '            torch.cuda.empty_cache()\n'
               '        return results\n'
               '\n'
               '    for task in tasks:\n'
               "        print(f'\\n--- {task.upper()} unpruned ---')\n"
               "        run_evaluation(ft_dir(task), task, 'DistilBERT unpruned', ft_dir(task))\n"
               '\n'
               "        print(f'\\n--- {task.upper()} pruned (60% sparsity) ---')\n"
               "        best = os.path.join(pr_dir(task), 'best')\n"
               '        if not model_saved(best):\n'
               "            print(f'  Pruned model not found at {best}. Run --block 2 --task {task} first.')\n"
               '        else:\n'
               "            run_evaluation(best, task, 'DistilBERT pruned 60%', pr_dir(task))\n"
               '\n'
               "    print('\\nBlock 3 done.')\n"
               '\n'
               '\n'
               '# ── Block 4: Results summary ───────────────────────────────────────────────────\n'
               'def block4():\n'
               "    header('BLOCK 4 — Full Results Summary (DistilBERT)')\n"
               '\n'
               '    metric_label = {\n'
               "        'sst2': 'Accuracy', 'qnli': 'Accuracy', 'mnli': 'Accuracy',\n"
               "        'qqp': 'F1', 'rte': 'Accuracy',\n"
               '    }\n'
               '\n'
               '    def load_result(path):\n'
               '        p = eval_path(path)\n'
               '        if not os.path.exists(p):\n'
               '            return None\n'
               '        return json.load(open(p))\n'
               '\n'
               '    W = 97\n'
               "    print('=' * W)\n"
               "    print('  COFI PRUNING RESULTS — DistilBERT on GLUE (60% Sparsity Target)')\n"
               "    print('=' * W)\n"
               '    print(f"  {\'Task\':<6} {\'Model\':<26} {\'Score\':>8} {\'Mem MB\':>9} "\n'
               '          f"{\'Lat ms\':>9} {\'Tput ex/s\':>11} {\'Sparsity\':>10}")\n'
               "    print('-' * W)\n"
               '\n'
               '    for task in ALL_TASKS:\n'
               '        ml  = metric_label[task]\n'
               '        unp = load_result(ft_dir(task))\n'
               '        pru = load_result(pr_dir(task))\n'
               '\n'
               '        def fmt(r):\n'
               '            if r is None:\n'
               "                return ['N/A'] * 5\n"
               "            score = r.get(ml, 'N/A')\n"
               '            return [\n'
               "                f'{score:.4f}' if isinstance(score, float) else str(score),\n"
               "                str(r.get('memory_mb', 'N/A')),\n"
               "                str(r.get('latency_ms', 'N/A')),\n"
               "                str(r.get('throughput_eps', 'N/A')),\n"
               '                f"{r.get(\'sparsity_pct\', \'N/A\')}%",\n'
               '            ]\n'
               '\n'
               '        u = fmt(unp)\n'
               '        p = fmt(pru)\n'
               '        print(f"  {task:<6} {\'DistilBERT unpruned\':<26} {u[0]:>8} {u[1]:>9} "\n'
               '              f"{u[2]:>9} {u[3]:>11} {u[4]:>10}")\n'
               '        print(f"  {\'\':<6} {\'DistilBERT pruned 60%\':<26} {p[0]:>8} {p[1]:>9} "\n'
               '              f"{p[2]:>9} {p[3]:>11} {p[4]:>10}")\n'
               '        if unp and pru:\n'
               '            try:\n'
               "                speedup   = unp['latency_ms'] / pru['latency_ms']\n"
               '                retention = float(p[0]) / float(u[0]) * 100\n'
               '                print(f"  {\'\':<6}   -> speedup {speedup:.2f}x   "\n'
               '                      f"score retention {retention:.1f}%")\n'
               '            except Exception:\n'
               '                pass\n'
               "        print('-' * W)\n"
               '\n'
               "    print('=' * W)\n"
               "    print('  Score: Accuracy for SST-2/QNLI/MNLI/RTE, F1 for QQP')\n"
               "    print('  Latency/Throughput: GPU, batch=32, 1000 validation examples')\n"
               "    print('  Sparsity: % reduction vs DistilBERT non-embedding params (~52M)')\n"
               '\n'
               '\n'
               '# ── Status ─────────────────────────────────────────────────────────────────────\n'
               'def show_status():\n'
               "    header('STATUS — DistilBERT')\n"
               '    W = 62\n'
               '    print(f"  {\'\':18}" + \'  \'.join(f\'{t:<6}\' for t in ALL_TASKS))\n'
               "    print('-' * W)\n"
               '    rows = [\n'
               "        ('FT downloaded', [('v' if model_saved(ft_dir(t)) else '.') for t in ALL_TASKS]),\n"
               "        ('Pruned',        [('v' if model_saved(os.path.join(pr_dir(t), 'best')) else '.') for t in "
               'ALL_TASKS]),\n'
               "        ('Eval unpruned', [('v' if os.path.exists(eval_path(ft_dir(t))) else '.') for t in "
               'ALL_TASKS]),\n'
               "        ('Eval pruned',   [('v' if os.path.exists(eval_path(pr_dir(t))) else '.') for t in "
               'ALL_TASKS]),\n'
               '    ]\n'
               '    for label, row in rows:\n'
               '        print(f"  {label:<18}" + \'  \'.join(f\'{s:<6}\' for s in row))\n'
               "    print('=' * W)\n"
               "    print('  v = done   . = not yet')\n"
               '\n'
               '\n'
               '# ── Main ───────────────────────────────────────────────────────────────────────\n'
               "if __name__ == '__main__':\n"
               '    parser = argparse.ArgumentParser(\n'
               "        description='CoFiPruning DistilBERT on GLUE')\n"
               "    parser.add_argument('--block', type=int, choices=[0, 1, 2, 3, 4])\n"
               "    parser.add_argument('--task',  type=str, choices=ALL_TASKS)\n"
               "    parser.add_argument('--status', action='store_true')\n"
               '    args = parser.parse_args()\n'
               '\n'
               '    tasks = [args.task] if args.task else ALL_TASKS\n'
               '\n'
               '    if args.status:\n'
               '        show_status()\n'
               '    elif args.block == 0:\n'
               '        block0()\n'
               '    elif args.block == 1:\n'
               '        block1(tasks)\n'
               '    elif args.block == 2:\n'
               '        block2(tasks)\n'
               '    elif args.block == 3:\n'
               '        block3(tasks)\n'
               '    elif args.block == 4:\n'
               '        block4()\n'
               '    else:\n'
               '        parser.print_help()\n',
 'mobilebert': '"""\n'
               'CoFiPruning — MobileBERT on 5 GLUE Tasks\n'
               '==========================================\n'
               'Run from the directory containing this script.\n'
               "All paths are relative to this script's location.\n"
               '\n'
               'MobileBERT architecture (google/mobilebert-uncased):\n'
               '  hidden_size = 512          (outer residual stream)\n'
               '  true_hidden_size = 128     (intra-bottleneck: attention Q/K/V, FFN)\n'
               '  intra_bottleneck_size = 128\n'
               '  intermediate_size = 512\n'
               '  num_attention_heads = 4\n'
               '  num_feedforward_networks = 4  (1 main + 3 extra FFN stacks per layer)\n'
               '  num_hidden_layers = 24\n'
               '  normalization_type = "no_norm"\n'
               '  use_bottleneck = True\n'
               '\n'
               'CoFi masking:\n'
               '  head_z / head_layer_z   → intra-bottleneck (128-dim)\n'
               '  intermediate_z / mlp_z  → shared across the 4 FFN stacks per layer\n'
               '  hidden_z                → outer stream (512-dim): position_embeddings,\n'
               '                            token_type_embeddings, embedding_transformation output,\n'
               '                            OutputBottleneck output, pooler, classifier input.\n'
               '                            NOT applied to word_embeddings (128-dim != 512).\n'
               '\n'
               'Block 0 writes modeling_mobilebert.py into CoFiPruning/models/ from the\n'
               'MODELING_MOBILEBERT_SOURCE string constant below.\n'
               'To update the modeling code: edit MODELING_MOBILEBERT_SOURCE and re-run block 0.\n'
               '\n'
               'Usage:\n'
               '  python cofi_mobilebert.py --status\n'
               '  python cofi_mobilebert.py --block 0           # fresh clone + all patches + write modeling file\n'
               '  python cofi_mobilebert.py --block 1           # download all 5 fine-tuned models\n'
               '  python cofi_mobilebert.py --block 1 --task rte\n'
               '  python cofi_mobilebert.py --block 2           # prune all 5\n'
               '  python cofi_mobilebert.py --block 2 --task rte\n'
               '  python cofi_mobilebert.py --block 3           # evaluate all\n'
               '  python cofi_mobilebert.py --block 3 --task rte\n'
               '  python cofi_mobilebert.py --block 4           # results table\n'
               '"""\n'
               '\n'
               'import argparse\n'
               'import json\n'
               'import os\n'
               'import re\n'
               'import subprocess\n'
               'import sys\n'
               'import time\n'
               '\n'
               '# ── Paths (relative — works on any server) ────────────────────────────────────\n'
               'BASE_DIR = os.path.dirname(os.path.abspath(__file__))\n'
               "MOBILE_REPO_DIR = os.path.join(BASE_DIR, 'CoFiPruning_Mobile')\n"
               '\n'
               '# ── Constants ──────────────────────────────────────────────────────────────────\n'
               'SPARSITY    = 0.6\n'
               'SEED        = 57\n'
               "ALL_TASKS   = ['sst2', 'qnli', 'mnli', 'qqp', 'rte']\n"
               "SMALL_TASKS = {'rte', 'mrpc', 'cola', 'stsb'}\n"
               '\n'
               '# Pre-finetuned MobileBERT checkpoints.\n'
               '# [MEDIUM] Names below are best-guess — verify on HuggingFace before running block 1.\n'
               '# If a name 404s, block1 will report it clearly. Search "mobilebert {task} glue".\n'
               'MOBILE_PRETRAINED_FT = {\n'
               "    'sst2': 'Alireza1044/mobilebert_sst2',          \n"
               "    'qnli': 'Alireza1044/mobilebert_QNLI',  \n"
               "    'mnli': 'Alireza1044/mobilebert_mnli',  \n"
               "    'qqp' : 'Alireza1044/mobilebert_qqp',   \n"
               "    'rte' : 'Alireza1044/mobilebert_rte',   \n"
               '}\n'
               '# [HIGH] The user MUST update MOBILE_PRETRAINED_FT with correct MobileBERT GLUE finetune\n'
               '# model IDs before running block 1. The placeholders above are WRONG.\n'
               '# Search HuggingFace for: "mobilebert sst2", "mobilebert qnli", etc.\n'
               '# Good starting point: search "mobilebert" on hf.co/models filtered by GLUE tasks.\n'
               '\n'
               '# Non-embedding param count — set to None to auto-compute in block3.\n'
               'MOBILEBERT_BASE_PARAMS = None\n'
               '\n'
               '# ── Path helpers ───────────────────────────────────────────────────────────────\n'
               'def ft_dir(task):\n'
               "    return os.path.join(BASE_DIR, f'ft_mobilebert_{task}')\n"
               '\n'
               'def pr_dir(task):\n'
               "    return os.path.join(BASE_DIR, f'pr_mobilebert_{task}_s{int(SPARSITY * 100)}')\n"
               '\n'
               'def eval_path(d):\n'
               "    return os.path.join(d, 'eval_results.json')\n"
               '\n'
               'def model_saved(path):\n'
               "    return (os.path.exists(os.path.join(path, 'pytorch_model.bin')) or\n"
               "            os.path.exists(os.path.join(path, 'model.safetensors')))\n"
               '\n'
               'def task_cfg(task):\n'
               '    if task in SMALL_TASKS:\n'
               '        return dict(prune_epochs=100, eval_steps=50,  save_steps=50,\n'
               '                    prepruning=4,  lag_warmup=20, layer_distill_v=4)\n'
               '    return dict(prune_epochs=20,  eval_steps=500, save_steps=500,\n'
               '                prepruning=1,  lag_warmup=2,  layer_distill_v=3)\n'
               '\n'
               'def header(msg):\n'
               '    W = 60\n'
               "    print('\\n' + '=' * W)\n"
               "    print(f'  {msg}')\n"
               "    print('=' * W)\n"
               '\n'
               'def patch_file(fpath, old, new, description):\n'
               '    """Idempotent string-replace patch."""\n'
               '    if not os.path.exists(fpath):\n'
               "        print(f'  [missing file]    {description}')\n"
               '        return\n'
               '    txt = open(fpath).read()\n'
               '    if new and new in txt:\n'
               "        print(f'  [already patched] {description}')\n"
               '        return\n'
               '    if old not in txt:\n'
               "        print(f'  [not found]       {description}')\n"
               '        return\n'
               "    open(fpath, 'w').write(txt.replace(old, new))\n"
               "    print(f'  [patched]         {description}')\n"
               '\n'
               '\n'
               '# ── MODELING SOURCE ────────────────────────────────────────────────────────────\n'
               '# Paste the full content of modeling_mobilebert.py here as a triple-quoted string.\n'
               '# Block 0 writes this string to CoFiPruning/models/modeling_mobilebert.py.\n'
               '# To update: edit this string and re-run block 0.\n'
               'MODELING_MOBILEBERT_SOURCE = """\n'
               '\n'
               '# CoFi-adapted MobileBERT for sequence classification.\n'
               '\n'
               '# Architecture (google/mobilebert-uncased):\n'
               '#   hidden_size = 512          — outer dimension (residual stream, pooler, classifier)\n'
               '#   true_hidden_size = 128     — intra-bottleneck size (attention Q/K/V, FFN up/down)\n'
               '#   intra_bottleneck_size = 128\n'
               '#   embedding_size = 128\n'
               '#   intermediate_size = 512    — FFN intermediate (note: same as hidden_size here)\n'
               '#   num_attention_heads = 4\n'
               '#   num_feedforward_networks = 4   — 1 main FFN + 3 extra FFNLayer stacks\n'
               '#   num_hidden_layers = 24\n'
               '#   normalization_type = "no_norm" — uses NoNorm, NOT LayerNorm\n'
               '#   use_bottleneck = True\n'
               '#   use_bottleneck_attention = False\n'
               '#   key_query_shared_bottleneck = True\n'
               '#   trigram_input = True\n'
               '\n'
               '# CoFi masking design decisions (per user):\n'
               '#   head_z / head_layer_z    — mask at true_hidden_size (128) granularity (intra-bottleneck)\n'
               "#   intermediate_z / mlp_z  — mask each FFN stack's intermediate dim independently\n"
               '#                              (4 stacks → 4 independent intermediate_z slices per layer)\n'
               '#                              These are indexed as intermediate_z[layer, :intermediate_size]\n'
               '#                              and the 4 FFN stacks share the SAME intermediate_z slice\n'
               '#                              (the stacks have identical structure, one mask is sufficient)\n'
               '#   hidden_z                 — masks hidden_size (512) — the OUTER residual dimension\n'
               '#                              applied to: position_embeddings, token_type_embeddings,\n'
               '#                              embedding_transformation output, OutputBottleneck output,\n'
               '#                              pooler, classifier input.\n'
               '#                              NOT applied to word_embeddings (embedding_size=128 ≠ 512).\n'
               '\n'
               '# NOTE on matrix dimensions in eval:\n'
               '#   After hidden_z prunes 512→N (e.g. 510):\n'
               '#     pooler.dense: Linear(512,512) → needs dim=1 AND dim=0 pruning\n'
               '#     classifier:   Linear(512, num_labels) → needs dim=1 pruning\n'
               '#     OutputBottleneck.dense: Linear(128,512) → needs dim=0 pruning\n'
               '#     position_embeddings: Embedding(512, 512) → index_select on dim=1\n'
               '#     token_type_embeddings: Embedding(2, 512) → index_select on dim=1\n'
               "#   After true_hidden_size pruning is NOT done (we don't prune intra-bottleneck dim\n"
               '#   independently — head pruning reduces true_hidden_size via prune_heads only).\n'
               '\n'
               '# Zero-head guard: if all heads pruned → return residual from bottleneck unchanged.\n'
               '# Zero-FFN guard: if ffn stack intermediate pruned to 0 → skip that stack.\n'
               '\n'
               '\n'
               'import math\n'
               'import logging\n'
               'from typing import Optional, Tuple, Union\n'
               '\n'
               'import torch\n'
               'import torch.nn as nn\n'
               'import torch.nn.functional as F\n'
               'from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss\n'
               '\n'
               'from transformers.models.mobilebert.modeling_mobilebert import (\n'
               '    MobileBertPreTrainedModel,\n'
               '    MobileBertEmbeddings,\n'
               '    MobileBertPooler,\n'
               '    MobileBertForSequenceClassification,\n'
               '    NoNorm,\n'
               '    NORM2FN,\n'
               ')\n'
               'from transformers.modeling_outputs import (\n'
               '    BaseModelOutput,\n'
               '    BaseModelOutputWithPooling,\n'
               '    SequenceClassifierOutput,\n'
               ')\n'
               'from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer\n'
               '\n'
               'logger = logging.getLogger(__name__)\n'
               '\n'
               '\n'
               '# ── CoFiNoNorm ─────────────────────────────────────────────────────────────────\n'
               'class CoFiNoNorm(nn.Module):\n'
               '    \n'
               '    # NoNorm that supports hidden_z masking (operates on whatever dim it was initialized with).\n'
               '    # MobileBERT uses NoNorm instead of LayerNorm in most places.\n'
               '    \n'
               '    def __init__(self, feat_size, eps=None):\n'
               '        super().__init__()\n'
               '        self.bias   = nn.Parameter(torch.zeros(feat_size))\n'
               '        self.weight = nn.Parameter(torch.ones(feat_size))\n'
               '        self.feat_size = feat_size\n'
               '\n'
               '    def forward(self, input_tensor, hidden_z=None):\n'
               '        # hidden_z is only meaningful when feat_size == hidden_size (512).\n'
               '        # For true_hidden_size (128) norms, hidden_z is None.\n'
               '        if hidden_z is not None and self.feat_size == hidden_z.shape[0]:\n'
               '            remaining = torch.where(~hidden_z.eq(0))[0]\n'
               '            compressed = torch.index_select(input_tensor, dim=-1, index=remaining)\n'
               '            out = compressed * self.weight[remaining] + self.bias[remaining]\n'
               '            result = input_tensor.clone()\n'
               '            result[:, :, remaining] = out\n'
               '            return result\n'
               '        return input_tensor * self.weight + self.bias\n'
               '\n'
               '\n'
               '# ── CoFiMobileBertSelfAttention ────────────────────────────────────────────────\n'
               'class CoFiMobileBertSelfAttention(nn.Module):\n'
               '    \n'
               '    # MobileBertSelfAttention with head_z masking.\n'
               '    # Operates at true_hidden_size (128) — intra-bottleneck dimension.\n'
               '    # Q/K input: true_hidden_size (128 with key_query_shared_bottleneck)\n'
               '    # V input: true_hidden_size (128 with use_bottleneck_attention=False, hidden_size otherwise)\n'
               '    \n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.num_attention_heads = config.num_attention_heads\n'
               '        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)\n'
               '        self.all_head_size       = self.num_attention_heads * self.attention_head_size\n'
               '\n'
               '        self.query   = nn.Linear(config.true_hidden_size, self.all_head_size)\n'
               '        self.key     = nn.Linear(config.true_hidden_size, self.all_head_size)\n'
               '        self.value   = nn.Linear(\n'
               '            config.true_hidden_size if config.use_bottleneck_attention else config.hidden_size,\n'
               '            self.all_head_size\n'
               '        )\n'
               '        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)\n'
               '\n'
               '    def transpose_for_scores(self, x):\n'
               '        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)\n'
               '        return x.view(new_shape).permute(0, 2, 1, 3)\n'
               '\n'
               '    def forward(self, query_tensor, key_tensor, value_tensor,\n'
               '                attention_mask=None, head_mask=None,\n'
               '                output_attentions=False, head_z=None):\n'
               '        # Zero-head guard\n'
               '        if self.value is None or self.num_attention_heads == 0:\n'
               '            return (None, None) if output_attentions else (None,)\n'
               '\n'
               '        q = self.transpose_for_scores(self.query(query_tensor))\n'
               '        k = self.transpose_for_scores(self.key(key_tensor))\n'
               '        v = self.transpose_for_scores(self.value(value_tensor))\n'
               '\n'
               '        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)\n'
               '        if attention_mask is not None:\n'
               '            scores = scores + attention_mask\n'
               '\n'
               '        probs   = F.softmax(scores, dim=-1)\n'
               '        probs   = self.dropout(probs)\n'
               '        if head_mask is not None:\n'
               '            probs = probs * head_mask\n'
               '\n'
               '        context = torch.matmul(probs, v)   # [bs, heads, seq, head_dim]\n'
               '\n'
               '        # Apply head_z: per-head gate\n'
               '        if head_z is not None:\n'
               '            context = context * head_z.view(1, self.num_attention_heads, 1, 1)\n'
               '\n'
               '        context = context.permute(0, 2, 1, 3).contiguous()\n'
               '        context = context.view(context.size()[:-2] + (self.all_head_size,))\n'
               '\n'
               '        return (context, probs) if output_attentions else (context,)\n'
               '\n'
               '    def prune_heads(self, heads):\n'
               '        if not heads:\n'
               '            return\n'
               '        heads, index = find_pruneable_heads_and_indices(\n'
               '            heads, self.num_attention_heads,\n'
               '            self.attention_head_size, set())\n'
               '        if len(index) == 0:\n'
               '            self.query = self.key = self.value = None\n'
               '            self.num_attention_heads = 0\n'
               '            self.all_head_size = 0\n'
               '        else:\n'
               '            self.query = prune_linear_layer(self.query, index)\n'
               '            self.key   = prune_linear_layer(self.key,   index)\n'
               '            self.value = prune_linear_layer(self.value, index)\n'
               '            self.num_attention_heads -= len(heads)\n'
               '            self.all_head_size = self.attention_head_size * self.num_attention_heads\n'
               '\n'
               '\n'
               '# ── CoFiMobileBertSelfOutput ───────────────────────────────────────────────────\n'
               'class CoFiMobileBertSelfOutput(nn.Module):\n'
               '    \n'
               '    # Post-attention projection within the intra-bottleneck (true_hidden_size → true_hidden_size).\n'
               '    # head_layer_z gates the entire attention output before residual.\n'
               '    # No hidden_z here — still in intra-bottleneck space (128-dim).\n'
               '    \n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.use_bottleneck = config.use_bottleneck\n'
               '        self.dense     = nn.Linear(config.true_hidden_size, config.true_hidden_size)\n'
               '        self.LayerNorm = CoFiNoNorm(config.true_hidden_size, eps=config.layer_norm_eps)\n'
               '        if not self.use_bottleneck:\n'
               '            self.dropout = nn.Dropout(config.hidden_dropout_prob)\n'
               '\n'
               '    def forward(self, hidden_states, residual_tensor,\n'
               '                head_layer_z=None):\n'
               '        if hidden_states is None:\n'
               '            # All heads pruned — skip projection, return residual as-is\n'
               '            return residual_tensor\n'
               '\n'
               '        out = self.dense(hidden_states)\n'
               '        if not self.use_bottleneck:\n'
               '            out = self.dropout(out)\n'
               '\n'
               '        if head_layer_z is not None:\n'
               '            out = out * head_layer_z\n'
               '\n'
               '        out = self.LayerNorm(out + residual_tensor)\n'
               '        return out\n'
               '\n'
               '\n'
               '# ── CoFiMobileBertAttention ────────────────────────────────────────────────────\n'
               'class CoFiMobileBertAttention(nn.Module):\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.self   = CoFiMobileBertSelfAttention(config)\n'
               '        self.output = CoFiMobileBertSelfOutput(config)\n'
               '\n'
               '    def forward(self, query_tensor, key_tensor, value_tensor, layer_input,\n'
               '                attention_mask=None, head_mask=None,\n'
               '                output_attentions=False, head_z=None, head_layer_z=None):\n'
               '        self_out = self.self(\n'
               '            query_tensor, key_tensor, value_tensor,\n'
               '            attention_mask=attention_mask,\n'
               '            head_mask=head_mask,\n'
               '            output_attentions=output_attentions,\n'
               '            head_z=head_z,\n'
               '        )\n'
               '        attn_output = self.output(self_out[0], layer_input,\n'
               '                                  head_layer_z=head_layer_z)\n'
               '        return (attn_output,) + self_out[1:]\n'
               '\n'
               '    def prune_heads(self, heads):\n'
               '        self.self.prune_heads(heads)\n'
               '\n'
               '\n'
               '# ── CoFiMobileBertIntermediate ─────────────────────────────────────────────────\n'
               'class CoFiMobileBertIntermediate(nn.Module):\n'
               '    # Up-projection: true_hidden_size (128) → intermediate_size (512).\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.dense = nn.Linear(config.true_hidden_size, config.intermediate_size)\n'
               '        self.act   = nn.ReLU()  # config.hidden_act = "relu"\n'
               '\n'
               '    def forward(self, hidden_states, intermediate_z=None):\n'
               '        out = self.act(self.dense(hidden_states))\n'
               '        if intermediate_z is not None:\n'
               '            out = out * intermediate_z\n'
               '        return out\n'
               '\n'
               '\n'
               '# ── CoFiOutputBottleneck ───────────────────────────────────────────────────────\n'
               'class CoFiOutputBottleneck(nn.Module):\n'
               '\n'
               '    # Projects from true_hidden_size (128) back to hidden_size (512).\n'
               '    # hidden_z masks the OUTPUT dimension (512-dim).\n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.dense     = nn.Linear(config.true_hidden_size, config.hidden_size)\n'
               '        self.LayerNorm = CoFiNoNorm(config.hidden_size, eps=config.layer_norm_eps)\n'
               '        self.dropout   = nn.Dropout(config.hidden_dropout_prob)\n'
               '\n'
               '    def forward(self, hidden_states, residual_tensor, mlp_z=None, hidden_z=None):\n'
               '        out = self.dense(hidden_states)\n'
               '        out = self.dropout(out)\n'
               '\n'
               '        if mlp_z is not None:\n'
               '            out = out * mlp_z\n'
               '\n'
               '        if hidden_z is not None:\n'
               '            out = out.mul(hidden_z)\n'
               '        out = self.LayerNorm(out + residual_tensor, hidden_z)\n'
               '        if hidden_z is not None:\n'
               '            out = out.mul(hidden_z)\n'
               '        return out\n'
               '\n'
               '\n'
               '# ── CoFiMobileBertOutput ───────────────────────────────────────────────────────\n'
               'class CoFiMobileBertOutput(nn.Module):\n'
               '    \n'
               '    # Down-projection: intermediate_size (512) → true_hidden_size (128),\n'
               '    # then OutputBottleneck projects to hidden_size (512).\n'
               '    # mlp_z gates the final bottleneck output (before residual add).\n'
               '    \n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.use_bottleneck = config.use_bottleneck\n'
               '        self.dense     = nn.Linear(config.intermediate_size, config.true_hidden_size)\n'
               '        self.LayerNorm = CoFiNoNorm(config.true_hidden_size)\n'
               '        if not self.use_bottleneck:\n'
               '            self.dropout = nn.Dropout(config.hidden_dropout_prob)\n'
               '        else:\n'
               '            self.bottleneck = CoFiOutputBottleneck(config)\n'
               '\n'
               '    def forward(self, intermediate_states, residual_tensor_1, residual_tensor_2,\n'
               '                mlp_z=None, hidden_z=None):\n'
               '        out = self.dense(intermediate_states)\n'
               '        if not self.use_bottleneck:\n'
               '            out = self.dropout(out)\n'
               '            out = self.LayerNorm(out + residual_tensor_1)\n'
               '        else:\n'
               '            out = self.LayerNorm(out + residual_tensor_1)\n'
               '            out = self.bottleneck(out, residual_tensor_2,\n'
               '                                  mlp_z=mlp_z, hidden_z=hidden_z)\n'
               '        return out\n'
               '\n'
               '\n'
               '# ── CoFiFFNOutput ──────────────────────────────────────────────────────────────\n'
               'class CoFiFFNOutput(nn.Module):\n'
               '    # Down-projection for extra FFN stacks: intermediate_size → true_hidden_size.\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.dense     = nn.Linear(config.intermediate_size, config.true_hidden_size)\n'
               '        self.LayerNorm = CoFiNoNorm(config.true_hidden_size, eps=config.layer_norm_eps)\n'
               '\n'
               '    def forward(self, hidden_states, residual_tensor, intermediate_z=None):\n'
               '        out = self.dense(hidden_states)\n'
               '        out = self.LayerNorm(out + residual_tensor)\n'
               '        return out\n'
               '\n'
               '\n'
               '# ── CoFiFFNLayer ───────────────────────────────────────────────────────────────\n'
               'class CoFiFFNLayer(nn.Module):\n'
               '    # One of the (num_feedforward_networks-1) extra FFN stacks per layer.\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.intermediate = CoFiMobileBertIntermediate(config)\n'
               '        self.output       = CoFiFFNOutput(config)\n'
               '\n'
               '    def forward(self, hidden_states, intermediate_z=None):\n'
               '        inter = self.intermediate(hidden_states, intermediate_z=intermediate_z)\n'
               '        # Zero-FFN guard: if all intermediate dims zeroed, skip\n'
               '        if inter.sum().eq(0).item():\n'
               '            return hidden_states\n'
               '        return self.output(inter, hidden_states)\n'
               '\n'
               '\n'
               '# ── CoFiBottleneckLayer ────────────────────────────────────────────────────────\n'
               'class CoFiBottleneckLayer(nn.Module):\n'
               '    # hidden_size (512) → intra_bottleneck_size (128).\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.dense     = nn.Linear(config.hidden_size, config.intra_bottleneck_size)\n'
               '        self.LayerNorm = CoFiNoNorm(config.intra_bottleneck_size, eps=config.layer_norm_eps)\n'
               '\n'
               '    def forward(self, hidden_states):\n'
               '        return self.LayerNorm(self.dense(hidden_states))\n'
               '\n'
               '\n'
               '# ── CoFiBottleneck ─────────────────────────────────────────────────────────────\n'
               'class CoFiBottleneck(nn.Module):\n'
               '    # Input bottleneck projections. No CoFi masks here — fixed infrastructure.\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck\n'
               '        self.use_bottleneck_attention    = config.use_bottleneck_attention\n'
               '        self.input = CoFiBottleneckLayer(config)\n'
               '        if self.key_query_shared_bottleneck:\n'
               '            self.attention = CoFiBottleneckLayer(config)\n'
               '\n'
               '    def forward(self, hidden_states):\n'
               '        bottlenecked = self.input(hidden_states)\n'
               '        if self.use_bottleneck_attention:\n'
               '            return (bottlenecked,) * 4\n'
               '        elif self.key_query_shared_bottleneck:\n'
               '            shared = self.attention(hidden_states)\n'
               '            return (shared, shared, hidden_states, bottlenecked)\n'
               '        else:\n'
               '            return (hidden_states, hidden_states, hidden_states, bottlenecked)\n'
               '\n'
               '\n'
               '# ── CoFiMobileBertLayer ────────────────────────────────────────────────────────\n'
               'class CoFiMobileBertLayer(nn.Module):\n'
               '\n'
               '    # Full MobileBERT transformer block with CoFi masking.\n'
               '\n'
               '    # Mask threading:\n'
               '    #   head_z          → SelfAttention (per-head gate, intra-bottleneck)\n'
               '    #   head_layer_z    → SelfOutput (attention output gate)\n'
               "    #   intermediate_z  → all FFN stacks' intermediate dims (shared mask across stacks)\n"
               '    #   mlp_z           → OutputBottleneck (gates the 128→512 projection output)\n'
               '    #   hidden_z        → OutputBottleneck + residual (512-dim outer stream)\n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.use_bottleneck          = config.use_bottleneck\n'
               '        self.num_feedforward_networks = config.num_feedforward_networks\n'
               '\n'
               '        self.attention    = CoFiMobileBertAttention(config)\n'
               '        self.intermediate = CoFiMobileBertIntermediate(config)\n'
               '        self.output       = CoFiMobileBertOutput(config)\n'
               '\n'
               '        if self.use_bottleneck:\n'
               '            self.bottleneck = CoFiBottleneck(config)\n'
               '\n'
               '        if config.num_feedforward_networks > 1:\n'
               '            self.ffn = nn.ModuleList(\n'
               '                [CoFiFFNLayer(config) for _ in range(config.num_feedforward_networks - 1)])\n'
               '        else:\n'
               '            self.ffn = nn.ModuleList([])\n'
               '\n'
               '    def forward(\n'
               '        self,\n'
               '        hidden_states,\n'
               '        attention_mask=None,\n'
               '        head_mask=None,\n'
               '        output_attentions=False,\n'
               '        head_z=None,\n'
               '        head_layer_z=None,\n'
               '        intermediate_z=None,\n'
               '        mlp_z=None,\n'
               '        hidden_z=None,\n'
               '    ):\n'
               '        if self.use_bottleneck:\n'
               '            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)\n'
               '        else:\n'
               '            query_tensor = key_tensor = value_tensor = layer_input = hidden_states\n'
               '\n'
               '        attn_outputs = self.attention(\n'
               '            query_tensor, key_tensor, value_tensor, layer_input,\n'
               '            attention_mask=attention_mask,\n'
               '            head_mask=head_mask,\n'
               '            output_attentions=output_attentions,\n'
               '            head_z=head_z,\n'
               '            head_layer_z=head_layer_z,\n'
               '        )\n'
               '        attention_output = attn_outputs[0]\n'
               '\n'
               '        # Extra FFN stacks (num_feedforward_networks - 1 of them)\n'
               '        for ffn_module in self.ffn:\n'
               '            attention_output = ffn_module(attention_output,\n'
               '                                          intermediate_z=intermediate_z)\n'
               '\n'
               '        # Main FFN: up-proj → (intermediate_z) → down-proj → OutputBottleneck\n'
               '        inter  = self.intermediate(attention_output, intermediate_z=intermediate_z)\n'
               '        output = self.output(inter, attention_output, hidden_states,\n'
               '                             mlp_z=mlp_z, hidden_z=hidden_z)\n'
               '\n'
               "        # Return (layer_output, [attentions], ...) matching original's tuple convention\n"
               '        # CoFiTrainer uses outputs[2] for hidden states — handled by encoder\n'
               '        outputs = (output,) + attn_outputs[1:]\n'
               '        return outputs\n'
               '\n'
               '\n'
               '# ── CoFiMobileBertEncoder ──────────────────────────────────────────────────────\n'
               'class CoFiMobileBertEncoder(nn.Module):\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.layer = nn.ModuleList(\n'
               '            [CoFiMobileBertLayer(config) for _ in range(config.num_hidden_layers)])\n'
               '\n'
               '    def forward(\n'
               '        self,\n'
               '        hidden_states,\n'
               '        attention_mask=None,\n'
               '        head_mask=None,\n'
               '        output_attentions=False,\n'
               '        output_hidden_states=False,\n'
               '        return_dict=True,\n'
               '        head_z=None,\n'
               '        head_layer_z=None,\n'
               '        intermediate_z=None,\n'
               '        mlp_z=None,\n'
               '        hidden_z=None,\n'
               '    ):\n'
               '        all_hidden_states = () if output_hidden_states else None\n'
               '        all_attentions    = () if output_attentions    else None\n'
               '\n'
               '        for i, layer_module in enumerate(self.layer):\n'
               '            if output_hidden_states:\n'
               '                all_hidden_states = all_hidden_states + (hidden_states,)\n'
               '\n'
               '            layer_outputs = layer_module(\n'
               '                hidden_states,\n'
               '                attention_mask=attention_mask,\n'
               '                head_mask=head_mask[i] if head_mask is not None else None,\n'
               '                output_attentions=output_attentions,\n'
               '                head_z=head_z[i]          if head_z          is not None else None,\n'
               '                head_layer_z=head_layer_z[i] if head_layer_z is not None else None,\n'
               '                intermediate_z=intermediate_z[i] if intermediate_z is not None else None,\n'
               '                mlp_z=mlp_z[i]            if mlp_z            is not None else None,\n'
               '                hidden_z=hidden_z,\n'
               '            )\n'
               '            hidden_states = layer_outputs[0]\n'
               '\n'
               '            if output_attentions:\n'
               '                all_attentions = all_attentions + (layer_outputs[1],)\n'
               '\n'
               '        if output_hidden_states:\n'
               '            all_hidden_states = all_hidden_states + (hidden_states,)\n'
               '\n'
               '        if not return_dict:\n'
               '            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions]\n'
               '                         if v is not None)\n'
               '        return BaseModelOutput(\n'
               '            last_hidden_state=hidden_states,\n'
               '            hidden_states=all_hidden_states,\n'
               '            attentions=all_attentions,\n'
               '        )\n'
               '\n'
               '\n'
               '# ── CoFiMobileBertModel ────────────────────────────────────────────────────────\n'
               'class CoFiMobileBertModel(nn.Module):\n'
               '    # MobileBertModel with CoFi masks threaded through encoder.\n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__()\n'
               '        self.config     = config\n'
               '        self.embeddings = MobileBertEmbeddings(config)   # unchanged\n'
               '        self.encoder    = CoFiMobileBertEncoder(config)\n'
               '        self.pooler     = MobileBertPooler(config)       # unchanged\n'
               '\n'
               '    def get_input_embeddings(self):\n'
               '        return self.embeddings.word_embeddings\n'
               '\n'
               '    def set_input_embeddings(self, value):\n'
               '        self.embeddings.word_embeddings = value\n'
               '\n'
               '    def get_extended_attention_mask(self, attention_mask, input_shape):\n'
               '        if attention_mask.dim() == 2:\n'
               '            ext = attention_mask[:, None, None, :]\n'
               '            ext = (1.0 - ext.to(dtype=next(self.parameters()).dtype)) * '
               'torch.finfo(torch.float32).min\n'
               '            return ext\n'
               '        return attention_mask\n'
               '\n'
               '    def forward(\n'
               '        self,\n'
               '        input_ids=None,\n'
               '        attention_mask=None,\n'
               '        token_type_ids=None,\n'
               '        position_ids=None,\n'
               '        head_mask=None,\n'
               '        inputs_embeds=None,\n'
               '        output_attentions=None,\n'
               '        output_hidden_states=None,\n'
               '        return_dict=None,\n'
               '        head_z=None,\n'
               '        head_layer_z=None,\n'
               '        intermediate_z=None,\n'
               '        mlp_z=None,\n'
               '        hidden_z=None,\n'
               '    ):\n'
               '        output_attentions    = output_attentions    if output_attentions    is not None else '
               'self.config.output_attentions\n'
               '        output_hidden_states = output_hidden_states if output_hidden_states is not None else '
               'self.config.output_hidden_states\n'
               '        return_dict          = return_dict          if return_dict          is not None else '
               'self.config.use_return_dict\n'
               '\n'
               '        if input_ids is not None and inputs_embeds is not None:\n'
               '            raise ValueError("Specify either input_ids or inputs_embeds, not both")\n'
               '        elif input_ids is not None:\n'
               '            input_shape = input_ids.size()\n'
               '            device = input_ids.device\n'
               '        elif inputs_embeds is not None:\n'
               '            input_shape = inputs_embeds.size()[:-1]\n'
               '            device = inputs_embeds.device\n'
               '        else:\n'
               '            raise ValueError("Specify input_ids or inputs_embeds")\n'
               '\n'
               '        if attention_mask  is None: attention_mask  = torch.ones(input_shape, device=device)\n'
               '        if token_type_ids  is None: token_type_ids  = torch.zeros(input_shape, dtype=torch.long, '
               'device=device)\n'
               '\n'
               '        extended_mask = self.get_extended_attention_mask(attention_mask, input_shape)\n'
               '        head_mask_full = [None] * self.config.num_hidden_layers if head_mask is None else head_mask\n'
               '\n'
               '        # Embeddings (no hidden_z here — word_embeddings is 128-dim, not 512-dim)\n'
               '        embedding_output = self.embeddings(\n'
               '            input_ids=input_ids,\n'
               '            token_type_ids=token_type_ids,\n'
               '            position_ids=position_ids,\n'
               '            inputs_embeds=inputs_embeds,\n'
               '        )\n'
               '\n'
               '        # hidden_z IS applied after embedding_transformation (which outputs hidden_size=512)\n'
               '        # MobileBertEmbeddings.forward already applies embedding_transformation internally,\n'
               '        # so its output is hidden_size=512. We gate it here.\n'
               '        if hidden_z is not None:\n'
               '            embedding_output = embedding_output.mul(hidden_z)\n'
               '\n'
               '        encoder_outputs = self.encoder(\n'
               '            embedding_output,\n'
               '            attention_mask=extended_mask,\n'
               '            head_mask=head_mask_full,\n'
               '            output_attentions=output_attentions,\n'
               '            output_hidden_states=output_hidden_states,\n'
               '            return_dict=return_dict,\n'
               '            head_z=head_z,\n'
               '            head_layer_z=head_layer_z,\n'
               '            intermediate_z=intermediate_z,\n'
               '            mlp_z=mlp_z,\n'
               '            hidden_z=hidden_z,\n'
               '        )\n'
               '\n'
               '        sequence_output = encoder_outputs[0]\n'
               '        pooled_output   = self.pooler(sequence_output) if self.pooler is not None else None\n'
               '\n'
               '        if not return_dict:\n'
               '            return (sequence_output, pooled_output) + encoder_outputs[1:]\n'
               '\n'
               '        return BaseModelOutputWithPooling(\n'
               '            last_hidden_state=sequence_output,\n'
               '            pooler_output=pooled_output,\n'
               '            hidden_states=encoder_outputs.hidden_states,\n'
               '            attentions=encoder_outputs.attentions,\n'
               '        )\n'
               '\n'
               '\n'
               '# ── CoFiMobileBertForSequenceClassification ────────────────────────────────────\n'
               'class CoFiMobileBertForSequenceClassification(MobileBertPreTrainedModel):\n'
               '\n'
               '    # CoFi-prunable MobileBERT for sequence classification.\n'
               '    # Mirrors CoFiBertForSequenceClassification from modeling_bert.py.\n'
               '\n'
               '    # Classifier: Linear(hidden_size=512, num_labels)\n'
               '    # Pooler:     MobileBertPooler — with classifier_activation=False (default),\n'
               '    #             this is just hidden_states[:,0] with no projection.\n'
               '    #             If classifier_activation=True, has a Linear(512,512) + tanh.\n'
               '\n'
               '    # After hidden_z pruning (512→N):\n'
               '    #   pooler.dense (if exists): needs dim=1 AND dim=0 pruning (square matrix)\n'
               '    #   classifier: needs dim=1 pruning (input dim pruned)\n'
               "    # This is handled in cofi_utils.py's prune_model_with_z hidden_z section.\n"
               '\n'
               '\n'
               '    def __init__(self, config):\n'
               '        super().__init__(config)\n'
               '        self.num_labels  = config.num_labels\n'
               '        self.config      = config\n'
               '        self.mobilebert  = CoFiMobileBertModel(config)\n'
               '\n'
               '        classifier_dropout = (\n'
               '            config.classifier_dropout\n'
               '            if config.classifier_dropout is not None\n'
               '            else config.hidden_dropout_prob\n'
               '        )\n'
               '        self.dropout    = nn.Dropout(classifier_dropout)\n'
               '        self.classifier = nn.Linear(config.hidden_size, config.num_labels)\n'
               '\n'
               "        self.do_layer_distill    = getattr(config, 'do_layer_distill', False)\n"
               '        self.layer_transformation = (\n'
               '            nn.Linear(config.hidden_size, config.hidden_size)\n'
               '            if self.do_layer_distill else None\n'
               '        )\n'
               '\n'
               '        self.post_init()\n'
               '\n'
               '    def forward(\n'
               '        self,\n'
               '        input_ids=None,\n'
               '        attention_mask=None,\n'
               '        token_type_ids=None,\n'
               '        position_ids=None,\n'
               '        head_mask=None,\n'
               '        inputs_embeds=None,\n'
               '        labels=None,\n'
               '        output_attentions=None,\n'
               '        output_hidden_states=None,\n'
               '        return_dict=None,\n'
               '        head_z=None,\n'
               '        head_layer_z=None,\n'
               '        intermediate_z=None,\n'
               '        mlp_z=None,\n'
               '        hidden_z=None,\n'
               '    ):\n'
               '        return_dict = return_dict if return_dict is not None else self.config.use_return_dict\n'
               '\n'
               '        outputs = self.mobilebert(\n'
               '            input_ids=input_ids,\n'
               '            attention_mask=attention_mask,\n'
               '            token_type_ids=token_type_ids,\n'
               '            position_ids=position_ids,\n'
               '            head_mask=head_mask,\n'
               '            inputs_embeds=inputs_embeds,\n'
               '            output_attentions=output_attentions,\n'
               '            output_hidden_states=True,   # always on for layer distillation\n'
               '            return_dict=return_dict,\n'
               '            head_z=head_z,\n'
               '            head_layer_z=head_layer_z,\n'
               '            intermediate_z=intermediate_z,\n'
               '            mlp_z=mlp_z,\n'
               '            hidden_z=hidden_z,\n'
               '        )\n'
               '\n'
               '        pooled_output = outputs[1]\n'
               '        pooled_output = self.dropout(pooled_output)\n'
               '        logits        = self.classifier(pooled_output)\n'
               '\n'
               '        loss = None\n'
               '        if labels is not None:\n'
               '            if self.config.problem_type is None:\n'
               '                if self.num_labels == 1:\n'
               "                    self.config.problem_type = 'regression'\n"
               '                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):\n'
               "                    self.config.problem_type = 'single_label_classification'\n"
               '                else:\n'
               "                    self.config.problem_type = 'multi_label_classification'\n"
               '\n'
               "            if self.config.problem_type == 'regression':\n"
               '                loss_fct = MSELoss()\n'
               '                loss = (loss_fct(logits.squeeze(), labels.squeeze())\n'
               '                        if self.num_labels == 1 else loss_fct(logits, labels))\n'
               "            elif self.config.problem_type == 'single_label_classification':\n"
               '                loss_fct = CrossEntropyLoss()\n'
               '                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))\n'
               "            elif self.config.problem_type == 'multi_label_classification':\n"
               '                loss_fct = BCEWithLogitsLoss()\n'
               '                loss = loss_fct(logits, labels)\n'
               '\n'
               '        if not return_dict:\n'
               '            output = (logits,) + outputs[2:]\n'
               '            return ((loss,) + output) if loss is not None else output\n'
               '\n'
               '        return SequenceClassifierOutput(\n'
               '            loss=loss,\n'
               '            logits=logits,\n'
               '            hidden_states=outputs.hidden_states,\n'
               '            attentions=outputs.attentions,\n'
               '        )\n'
               '"""\n'
               '\n'
               '\n'
               '# ── Block 0: Fresh clone + all patches + write modeling file ──────────────────\n'
               '# ── Block 0: Fresh clone + all patches + write modeling file ──────────────────\n'
               'def block0():\n'
               "    header('BLOCK 0 — Fresh clone + patch CoFiPruning for MobileBERT')\n"
               '\n'
               "    if MODELING_MOBILEBERT_SOURCE.strip() == 'PASTE THE FULL CONTENT OF modeling_mobilebert.py HERE':\n"
               "        print('ERROR: MODELING_MOBILEBERT_SOURCE is still a placeholder.')\n"
               "        print('Paste the content of modeling_mobilebert.py into the string in this script.')\n"
               '        sys.exit(1)\n'
               '\n'
               '    # ── Fresh clone ────────────────────────────────────────────────────────────\n'
               "    print('\\n  Refreshing repo...')\n"
               '    if os.path.exists(MOBILE_REPO_DIR):\n'
               "        subprocess.run(['chmod', '-R', 'u+w', MOBILE_REPO_DIR], check=True)\n"
               "        subprocess.run(['rm', '-rf', MOBILE_REPO_DIR], check=True)\n"
               "        print(f'  Removed old {MOBILE_REPO_DIR}')\n"
               '    subprocess.run(\n'
               "        ['git', 'clone', 'https://github.com/princeton-nlp/CoFiPruning.git', MOBILE_REPO_DIR],\n"
               '        check=True)\n'
               "    print(f'  Cloned fresh into {MOBILE_REPO_DIR}')\n"
               '\n'
               '    # ── Write modeling_mobilebert.py ───────────────────────────────────────────\n'
               "    modeling_dst = os.path.join(MOBILE_REPO_DIR, 'models', 'modeling_mobilebert.py')\n"
               "    with open(modeling_dst, 'w') as f:\n"
               '        f.write(MODELING_MOBILEBERT_SOURCE)\n'
               "    print(f'  [written]         models/modeling_mobilebert.py '\n"
               "          f'({len(MODELING_MOBILEBERT_SOURCE.splitlines())} lines)')\n"
               '\n'
               '    # ── Patch modeling_mobilebert.py ───────────────────────────────────────────\n'
               "    print('\\n  Patching models/modeling_mobilebert.py...')\n"
               "    mb_path = os.path.join(MOBILE_REPO_DIR, 'models', 'modeling_mobilebert.py')\n"
               '\n'
               '    # 1. CoFiMobileBertIntermediate.forward() — None guard for self.dense\n'
               '    patch_file(mb_path,\n'
               "        '    def forward(self, hidden_states, intermediate_z=None):\\n'\n"
               "        '        out = self.act(self.dense(hidden_states))\\n'\n"
               "        '        if intermediate_z is not None:\\n'\n"
               "        '            out = out * intermediate_z\\n'\n"
               "        '        return out',\n"
               "        '    def forward(self, hidden_states, intermediate_z=None):\\n'\n"
               "        '        if self.dense is None:\\n'\n"
               "        '            return None\\n'\n"
               "        '        out = self.act(self.dense(hidden_states))\\n'\n"
               "        '        if intermediate_z is not None:\\n'\n"
               "        '            out = out * intermediate_z\\n'\n"
               "        '        return out',\n"
               "        'modeling_mobilebert: CoFiMobileBertIntermediate None guard')\n"
               '\n'
               '    # 2. CoFiFFNLayer.forward() — update zero-FFN guard to handle None\n'
               '    patch_file(mb_path,\n'
               "        '        inter = self.intermediate(hidden_states, intermediate_z=intermediate_z)\\n'\n"
               "        '        # Zero-FFN guard: if all intermediate dims zeroed, skip\\n'\n"
               "        '        if inter.sum().eq(0).item():\\n'\n"
               "        '            return hidden_states\\n'\n"
               "        '        return self.output(inter, hidden_states)',\n"
               "        '        inter = self.intermediate(hidden_states, intermediate_z=intermediate_z)\\n'\n"
               "        '        # Zero-FFN guard: if intermediate is None or all dims zeroed, skip\\n'\n"
               "        '        if inter is None or inter.sum().eq(0).item():\\n'\n"
               "        '            return hidden_states\\n'\n"
               "        '        return self.output(inter, hidden_states)',\n"
               "        'modeling_mobilebert: CoFiFFNLayer None guard')\n"
               '\n'
               '    # 3. CoFiBottleneck.forward() — handle self.attention is None\n'
               '    patch_file(mb_path,\n'
               "        '        elif self.key_query_shared_bottleneck:\\n'\n"
               "        '            shared = self.attention(hidden_states)\\n'\n"
               "        '            return (shared, shared, hidden_states, bottlenecked)',\n"
               "        '        elif self.key_query_shared_bottleneck:\\n'\n"
               "        '            if self.attention is None:\\n'\n"
               "        '                # All heads pruned for this layer — query/key branch unused\\n'\n"
               "        '                return (None, None, hidden_states, bottlenecked)\\n'\n"
               "        '            shared = self.attention(hidden_states)\\n'\n"
               "        '            return (shared, shared, hidden_states, bottlenecked)',\n"
               "        'modeling_mobilebert: CoFiBottleneck attention None guard')\n"
               '\n'
               '    # 4. CoFiMobileBertAttention.prune_heads() — add output.dense dim=1 pruning\n'
               '    patch_file(mb_path,\n'
               "        '    def prune_heads(self, heads):\\n'\n"
               "        '        self.self.prune_heads(heads)',\n"
               "        '    def prune_heads(self, heads):\\n'\n"
               "        '        if not heads:\\n'\n"
               "        '            return\\n'\n"
               "        '        from transformers.pytorch_utils import find_pruneable_heads_and_indices, "
               "prune_linear_layer\\n'\n"
               "        '        old_num_heads = self.self.num_attention_heads\\n'\n"
               "        '        old_head_size = self.self.attention_head_size\\n'\n"
               "        '        heads_set, index = find_pruneable_heads_and_indices(\\n'\n"
               "        '            heads, old_num_heads, old_head_size, set())\\n'\n"
               "        '        self.self.prune_heads(heads)\\n'\n"
               "        '        if len(index) == 0:\\n'\n"
               "        '            self.output.dense = None\\n'\n"
               "        '        else:\\n'\n"
               "        '            self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)',\n"
               "        'modeling_mobilebert: CoFiMobileBertAttention.prune_heads output.dense')\n"
               '\n'
               '    # 5. CoFiMobileBertLayer.forward() — inter is None pass-through\n'
               '    patch_file(mb_path,\n'
               "        '        # Main FFN: up-proj → (intermediate_z) → down-proj → OutputBottleneck\\n'\n"
               "        '        inter  = self.intermediate(attention_output, intermediate_z=intermediate_z)\\n'\n"
               "        '        output = self.output(inter, attention_output, hidden_states,\\n'\n"
               "        '                             mlp_z=mlp_z, hidden_z=hidden_z)',\n"
               "        '        # Main FFN: up-proj → (intermediate_z) → down-proj → OutputBottleneck\\n'\n"
               "        '        inter  = self.intermediate(attention_output, intermediate_z=intermediate_z)\\n'\n"
               "        '        if inter is None:\\n'\n"
               "        '            # Whole main FFN pruned — route through OutputBottleneck for residual\\n'\n"
               "        '            output = self.output.bottleneck(attention_output, hidden_states,\\n'\n"
               "        '                                            mlp_z=mlp_z, hidden_z=hidden_z) \\\\\\n'\n"
               '        \'                     if hasattr(self.output, "bottleneck") else attention_output\\n\'\n'
               "        '        else:\\n'\n"
               "        '            output = self.output(inter, attention_output, hidden_states,\\n'\n"
               "        '                                 mlp_z=mlp_z, hidden_z=hidden_z)',\n"
               "        'modeling_mobilebert: CoFiMobileBertLayer inter None pass-through')\n"
               '\n'
               '    # 6. CoFiMobileBertModel — add real _prune_heads\n'
               '    patch_file(mb_path,\n'
               "        '    def get_input_embeddings(self):\\n'\n"
               "        '        return self.embeddings.word_embeddings',\n"
               "        '    def _prune_heads(self, heads_to_prune):\\n'\n"
               "        '        for layer, heads in heads_to_prune.items():\\n'\n"
               "        '            self.encoder.layer[layer].attention.prune_heads(heads)\\n'\n"
               "        '\\n'\n"
               "        '    def get_input_embeddings(self):\\n'\n"
               "        '        return self.embeddings.word_embeddings',\n"
               "        'modeling_mobilebert: CoFiMobileBertModel._prune_heads real implementation')\n"
               '\n'
               '    # 7. output_hidden_states conditional (OOM fix)\n'
               '    patch_file(mb_path,\n'
               "        '            output_hidden_states=True,   # always on for layer distillation\\n',\n"
               "        '            output_hidden_states=(True if self.training else output_hidden_states),\\n',\n"
               "        'modeling_mobilebert: output_hidden_states conditional — OOM fix')\n"
               '\n'
               '    # ── Standard import patches (all .py files) ────────────────────────────────\n'
               "    print('\\n  Standard import patches...')\n"
               '    IMPORT_PATCHES = [\n'
               "        ('from transformers.file_utils import hf_bucket_url, cached_path',\n"
               "         'from huggingface_hub import cached_download as cached_path'),\n"
               "        ('from transformers.file_utils import cached_path',\n"
               "         'from huggingface_hub import cached_download as cached_path'),\n"
               "        ('from transformers.file_utils import hf_bucket_url', ''),\n"
               "        ('from datasets import load_dataset, load_metric, DatasetDict',\n"
               "         'from datasets import load_dataset, DatasetDict\\nimport evaluate'),\n"
               '        (\'metric = load_metric("glue", data_args.task_name)\',\n'
               '         \'metric = evaluate.load("glue", data_args.task_name)\'),\n'
               '        (\'metric = load_metric("accuracy")\',\n'
               '         \'metric = evaluate.load("accuracy")\'),\n'
               "        ('from black import main', ''),\n"
               '    ]\n'
               '    for root, dirs, files in os.walk(MOBILE_REPO_DIR):\n'
               "        dirs[:] = [d for d in dirs if d != '.git']\n"
               '        for fname in files:\n'
               "            if not fname.endswith('.py'):\n"
               '                continue\n'
               '            fpath = os.path.join(root, fname)\n'
               '            rel   = os.path.relpath(fpath, MOBILE_REPO_DIR)\n'
               '            for old, new in IMPORT_PATCHES:\n'
               '                patch_file(fpath, old, new, rel)\n'
               '\n'
               '    # ── run_glue_prune.py ──────────────────────────────────────────────────────\n'
               "    print('\\n  Patching run_glue_prune.py...')\n"
               "    glue_path = os.path.join(MOBILE_REPO_DIR, 'run_glue_prune.py')\n"
               '\n'
               '    patch_file(glue_path,\n'
               '        \'load_dataset("glue", data_args.task_name)\',\n'
               '        \'load_dataset("glue", data_args.task_name, trust_remote_code=True)\',\n'
               "        'add trust_remote_code')\n"
               '    patch_file(glue_path,\n'
               '        \'"evaluation_strategy"\',\n'
               '        \'"eval_strategy"\',\n'
               "        'evaluation_strategy -> eval_strategy')\n"
               '\n'
               '    patch_file(glue_path,\n'
               "        'from models.modeling_bert import CoFiBertForSequenceClassification\\n'\n"
               "        'from models.modeling_roberta import CoFiRobertaForSequenceClassification',\n"
               "        'from models.modeling_bert import CoFiBertForSequenceClassification\\n'\n"
               "        'from models.modeling_roberta import CoFiRobertaForSequenceClassification\\n'\n"
               "        'from models.modeling_mobilebert import CoFiMobileBertForSequenceClassification',\n"
               "        'run_glue_prune.py: import CoFiMobileBertForSequenceClassification')\n"
               '\n'
               '    patch_file(glue_path,\n'
               "        '    Model = CoFiBertForSequenceClassification if "
               "model_args.model_name_or_path.startswith(\\n'\n"
               '        \'        "bert") else CoFiRobertaForSequenceClassification\',\n'
               '        \'    if "mobilebert" in model_args.model_name_or_path.lower():\\n\'\n'
               "        '        Model = CoFiMobileBertForSequenceClassification\\n'\n"
               '        \'    elif "distilbert" in model_args.model_name_or_path.lower():\\n\'\n'
               "        '        from models.modeling_distilbert import CoFiDistilBertForSequenceClassification\\n'\n"
               "        '        Model = CoFiDistilBertForSequenceClassification\\n'\n"
               '        \'    elif (model_args.model_name_or_path.startswith("bert")\\n\'\n'
               "        '          or os.path.exists(os.path.join(model_args.model_name_or_path, "
               '"config.json"))):\\n\'\n'
               "        '        Model = CoFiBertForSequenceClassification\\n'\n"
               "        '    else:\\n'\n"
               "        '        Model = CoFiRobertaForSequenceClassification',\n"
               "        'run_glue_prune.py: model selection — mobilebert-aware')\n"
               '\n'
               '    # ── models/modeling_bert.py: remove broken from_pretrained override ────────\n'
               "    print('\\n  Patching models/modeling_bert.py...')\n"
               "    bert_path = os.path.join(MOBILE_REPO_DIR, 'models', 'modeling_bert.py')\n"
               '    src = open(bert_path).read()\n'
               '    pat = re.compile(\n'
               "        r'[ \\t]*@classmethod\\s*\\n[ \\t]*def from_pretrained\\(cls.*?(?=\\n[ \\t]{0,4}(?:def |class "
               "|\\Z))',\n"
               '        re.DOTALL)\n'
               '    if re.search(pat, src):\n'
               "        open(bert_path, 'w').write(re.sub(pat, '', src))\n"
               "        print('  [patched]         modeling_bert.py: removed from_pretrained override')\n"
               '    else:\n'
               "        print('  [already patched] modeling_bert.py')\n"
               '\n'
               '    # ── utils/cofi_utils.py ────────────────────────────────────────────────────\n'
               "    print('\\n  Patching utils/cofi_utils.py...')\n"
               "    utils_path = os.path.join(MOBILE_REPO_DIR, 'utils', 'cofi_utils.py')\n"
               '\n'
               '    # safetensors support\n'
               '    patch_file(utils_path,\n'
               '        \'    p = os.path.join(model_path, "pytorch_model.bin")\\n\'\n'
               '        \'    loaded_weights = torch.load(p, map_location="cpu")\',\n'
               '        \'    p_bin  = os.path.join(model_path, "pytorch_model.bin")\\n\'\n'
               '        \'    p_safe = os.path.join(model_path, "model.safetensors")\\n\'\n'
               "        '    if os.path.exists(p_bin):\\n'\n"
               '        \'        loaded_weights = torch.load(p_bin, map_location="cpu")\\n\'\n'
               "        '    elif os.path.exists(p_safe):\\n'\n"
               "        '        from safetensors.torch import load_file\\n'\n"
               "        '        loaded_weights = load_file(p_safe)\\n'\n"
               "        '    else:\\n'\n"
               '        \'        raise FileNotFoundError(f"No weights found in {model_path}")\',\n'
               "        'cofi_utils: safetensors support')\n"
               '\n'
               '    # _get_layers helper\n'
               '    patch_file(utils_path,\n'
               "        'def prune_model_with_z(zs, model):',\n"
               "        'def _get_layers(bert):\\n'\n"
               '        \'    """Return list of transformer layer objects, model-agnostically."""\\n\'\n'
               '        \'    if hasattr(bert, "encoder") and hasattr(bert.encoder, "layer"):\\n\'\n'
               "        '        return bert.encoder.layer\\n'\n"
               '        \'    if hasattr(bert, "transformer"):  # DistilBERT\\n\'\n'
               "        '        return bert.transformer.layer\\n'\n"
               '        \'    raise ValueError(f"Cannot find layers in {type(bert)}")\\n\\n\'\n'
               "        'def prune_model_with_z(zs, model):',\n"
               "        'cofi_utils: _get_layers helper')\n"
               '\n'
               '    # prune_model_with_z model type detection\n'
               '    patch_file(utils_path,\n'
               "        'def prune_model_with_z(zs, model):\\n'\n"
               "        '    if zs is None:\\n'\n"
               "        '        return None, None\\n'\n"
               '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\',\n'
               "        'def prune_model_with_z(zs, model):\\n'\n"
               "        '    if zs is None:\\n'\n"
               "        '        return None, None\\n'\n"
               '        \'    if hasattr(model, "mobilebert"):\\n\'\n'
               "        '        bert = model.mobilebert\\n'\n"
               "        '        num_layers = model.config.num_hidden_layers\\n'\n"
               "        '        is_mobilebert = True\\n'\n"
               '        \'    elif hasattr(model, "bert"):\\n\'\n'
               "        '        bert = model.bert\\n'\n"
               "        '        num_layers = model.config.num_hidden_layers\\n'\n"
               "        '        is_mobilebert = False\\n'\n"
               '        \'    elif hasattr(model, "roberta"):\\n\'\n'
               "        '        bert = model.roberta\\n'\n"
               "        '        num_layers = model.config.num_hidden_layers\\n'\n"
               "        '        is_mobilebert = False\\n'\n"
               '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
               "        '        bert = model.distilbert\\n'\n"
               "        '        num_layers = model.config.n_layers\\n'\n"
               "        '        is_mobilebert = False\\n'\n"
               "        '    else:\\n'\n"
               '        \'        raise ValueError(f"Unknown model type: {type(model)}")\',\n'
               "        'cofi_utils: prune_model_with_z model type detection')\n"
               '\n'
               '    # hidden_z layer loop — mobilebert/bert/distilbert\n'
               '    patch_file(utils_path,\n'
               "        '        for layer in range(0, 12):\\n'\n"
               "        '            if bert.encoder.layer[layer].attention.self.query is not None:\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.query = \\\\\\n'\n"
               "        '                    prune_layer(bert.encoder.layer[layer].attention.self.query , index, "
               "dim=1)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.key = \\\\\\n'\n"
               "        '                    prune_layer(bert.encoder.layer[layer].attention.self.key , index, "
               "dim=1)\\n'\n"
               "        '            if bert.encoder.layer[layer].attention.self.value is not None:\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.value = \\\\\\n'\n"
               "        '                    prune_layer(bert.encoder.layer[layer].attention.self.value , index, "
               "dim=1)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.output.dense = \\\\\\n'\n"
               "        '                    prune_layer(bert.encoder.layer[layer].attention.output.dense , index, "
               "dim=0)\\n'\n"
               "        '                prune_layer_norm(bert.encoder.layer[layer].attention.output.LayerNorm, "
               "index)\\n'\n"
               "        '            if bert.encoder.layer[layer].intermediate.dense is not None:\\n'\n"
               "        '                bert.encoder.layer[layer].intermediate.dense = \\\\\\n'\n"
               "        '                    prune_layer( bert.encoder.layer[layer].intermediate.dense, index, "
               "dim=1)\\n'\n"
               "        '                bert.encoder.layer[layer].output.dense = \\\\\\n'\n"
               "        '                    prune_layer( bert.encoder.layer[layer].output.dense, index, dim=0)\\n'\n"
               "        '                prune_layer_norm(bert.encoder.layer[layer].output.LayerNorm, index)',\n"
               "        '        layers = _get_layers(bert)\\n'\n"
               '        \'        _num_layers_print = getattr(model.config, "num_hidden_layers", 12)\\n\'\n'
               "        '        for layer in range(_num_layers_print):\\n'\n"
               "        '            lyr = layers[layer]\\n'\n"
               "        '            if is_mobilebert:\\n'\n"
               "        '                # MobileBERT hidden_z (512-dim) applies to OutputBottleneck + input "
               "Bottleneck\\n'\n"
               '        \'                if hasattr(lyr.output, "bottleneck"):\\n\'\n'
               "        '                    bn = lyr.output.bottleneck\\n'\n"
               "        '                    bn.dense = prune_layer(bn.dense, index, dim=0)\\n'\n"
               "        '                    prune_layer_norm(bn.LayerNorm, index)\\n'\n"
               '        \'                if hasattr(lyr, "bottleneck"):\\n\'\n'
               "        '                    in_bn = lyr.bottleneck\\n'\n"
               "        '                    in_bn.input.dense = prune_layer(in_bn.input.dense, index, dim=1)\\n'\n"
               '        \'                    if hasattr(in_bn, "attention") and in_bn.attention is not None:\\n\'\n'
               "        '                        in_bn.attention.dense = prune_layer(in_bn.attention.dense, index, "
               "dim=1)\\n'\n"
               "        '                # value takes hidden_size as input when use_bottleneck_attention=False\\n'\n"
               "        '                if lyr.attention.self.value is not None and "
               "lyr.attention.self.value.in_features == hidden_zs.shape[0]:\\n'\n"
               "        '                    lyr.attention.self.value = prune_layer(lyr.attention.self.value, index, "
               "dim=1)\\n'\n"
               '        \'            elif hasattr(lyr, "attention") and hasattr(lyr.attention, "self"):\\n\'\n'
               "        '                # BERT-style\\n'\n"
               "        '                if lyr.attention.self.query is not None:\\n'\n"
               "        '                    lyr.attention.self.query = prune_layer(lyr.attention.self.query, index, "
               "dim=1)\\n'\n"
               "        '                    lyr.attention.self.key   = prune_layer(lyr.attention.self.key,   index, "
               "dim=1)\\n'\n"
               "        '                if lyr.attention.self.value is not None:\\n'\n"
               "        '                    lyr.attention.self.value = prune_layer(lyr.attention.self.value, index, "
               "dim=1)\\n'\n"
               "        '                    lyr.attention.output.dense = prune_layer(lyr.attention.output.dense, "
               "index, dim=0)\\n'\n"
               "        '                    prune_layer_norm(lyr.attention.output.LayerNorm, index)\\n'\n"
               "        '                if lyr.intermediate.dense is not None:\\n'\n"
               "        '                    lyr.intermediate.dense = prune_layer(lyr.intermediate.dense, index, "
               "dim=1)\\n'\n"
               "        '                    lyr.output.dense       = prune_layer(lyr.output.dense,       index, "
               "dim=0)\\n'\n"
               "        '                    prune_layer_norm(lyr.output.LayerNorm, index)\\n'\n"
               '        \'            elif hasattr(lyr, "attention") and hasattr(lyr.attention, "q_lin"):\\n\'\n'
               "        '                # DistilBERT-style\\n'\n"
               "        '                if lyr.attention.q_lin is not None:\\n'\n"
               "        '                    lyr.attention.q_lin  = prune_layer(lyr.attention.q_lin,  index, "
               "dim=1)\\n'\n"
               "        '                    lyr.attention.k_lin  = prune_layer(lyr.attention.k_lin,  index, "
               "dim=1)\\n'\n"
               "        '                if lyr.attention.v_lin is not None:\\n'\n"
               "        '                    lyr.attention.v_lin   = prune_layer(lyr.attention.v_lin,   index, "
               "dim=1)\\n'\n"
               "        '                    lyr.attention.out_lin = prune_layer(lyr.attention.out_lin, index, "
               "dim=0)\\n'\n"
               "        '                    prune_layer_norm(lyr.attn_output.LayerNorm, index)\\n'\n"
               '        \'                if hasattr(lyr, "ffn_lin1") and lyr.ffn_lin1 is not None:\\n\'\n'
               "        '                    lyr.ffn_lin1 = prune_layer(lyr.ffn_lin1, index, dim=1)\\n'\n"
               "        '                    lyr.ffn_lin2 = prune_layer(lyr.ffn_lin2, index, dim=0)\\n'\n"
               "        '                    prune_layer_norm(lyr.ffn_output.LayerNorm, index)',\n"
               "        'cofi_utils: hidden_z layer loop — mobilebert/bert/distilbert')\n"
               '\n'
               '    # hidden_z embeddings index_select — skip word_embeddings for mobilebert\n'
               '    patch_file(utils_path,\n'
               "        '        bert.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
               "        '            bert.embeddings.word_embeddings.weight.index_select(1, "
               "index).clone().detach())\\n'\n"
               "        '        bert.embeddings.word_embeddings.embedding_dim = index.shape[0]\\n'\n"
               "        '        bert.embeddings.position_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
               "        '            bert.embeddings.position_embeddings.weight.index_select(1, "
               "index).clone().detach())\\n'\n"
               "        '        bert.embeddings.position_embeddings.embedding_dim = index.shape[0]\\n'\n"
               "        '        bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
               "        '            bert.embeddings.token_type_embeddings.weight.index_select(1, "
               "index).clone().detach())\\n'\n"
               "        '        bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]\\n'\n"
               "        '        prune_layer_norm(bert.embeddings.LayerNorm, index)',\n"
               "        '        # MobileBERT: word_embeddings is 128-dim (skip), position+token_type are 512-dim\\n'\n"
               '        \'        if not hasattr(model, "mobilebert"):\\n\'\n'
               "        '            bert.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
               "        '                bert.embeddings.word_embeddings.weight.index_select(1, "
               "index).clone().detach())\\n'\n"
               "        '            bert.embeddings.word_embeddings.embedding_dim = index.shape[0]\\n'\n"
               "        '        bert.embeddings.position_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
               "        '            bert.embeddings.position_embeddings.weight.index_select(1, "
               "index).clone().detach())\\n'\n"
               "        '        bert.embeddings.position_embeddings.embedding_dim = index.shape[0]\\n'\n"
               '        \'        if hasattr(bert.embeddings, "token_type_embeddings"):\\n\'\n'
               "        '            bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
               "        '                bert.embeddings.token_type_embeddings.weight.index_select(1, "
               "index).clone().detach())\\n'\n"
               "        '            bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]\\n'\n"
               "        '        # MobileBERT: embedding_transformation output is hidden_size-dim — prune it\\n'\n"
               '        \'        if hasattr(model, "mobilebert") and hasattr(bert.embeddings, '
               '"embedding_transformation"):\\n\'\n'
               "        '            bert.embeddings.embedding_transformation = prune_linear_layer(\\n'\n"
               "        '                bert.embeddings.embedding_transformation, index, dim=0)\\n'\n"
               "        '            # Prune embedding LayerNorm (NoNorm) — guard against already-pruned case\\n'\n"
               "        '            try:\\n'\n"
               "        '                prune_layer_norm(bert.embeddings.LayerNorm, index)\\n'\n"
               "        '            except (IndexError, RuntimeError):\\n'\n"
               "        '                pass  # NoNorm already pruned or size mismatch — safe to skip\\n'\n"
               "        '        else:\\n'\n"
               "        '            prune_layer_norm(bert.embeddings.LayerNorm, index)',\n"
               "        'cofi_utils: hidden_z embeddings — skip word_embeddings for mobilebert + "
               "embedding_transformation')\n"
               '\n'
               '    # classifier/pooler — mobilebert + distilbert + bert\n'
               '    patch_file(utils_path,\n'
               '        \'        if hasattr(model, "classifier"):\\n\'\n'
               '        \'            if hasattr(model.classifier, "dense"):\\n\'\n'
               "        '                model.classifier.dense = prune_linear_layer(model.classifier.dense, index, "
               "dim=1)\\n'\n"
               '        \'        if hasattr(model, "cls"):\\n\'\n'
               '        \'            if hasattr(model.cls, "dense"):\\n\'\n'
               "        '                model.cls.dense = prune_linear_layer(model.classifier.dense, index, "
               "dim=1)\\n'\n"
               '        \'        if hasattr(bert.pooler, "dense"):\\n\'\n'
               "        '            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)\\n'\n"
               '        \'        if hasattr(model, "qa_outputs"):\\n\'\n'
               "        '            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)',\n"
               '        \'        if hasattr(model, "pre_classifier"):\\n\'\n'
               "        '            model.pre_classifier = prune_linear_layer(model.pre_classifier, index, "
               "dim=1)\\n'\n"
               "        '            model.pre_classifier = prune_linear_layer(model.pre_classifier, index, "
               "dim=0)\\n'\n"
               '        \'        if hasattr(model, "classifier"):\\n\'\n'
               '        \'            if hasattr(model.classifier, "dense"):\\n\'\n'
               "        '                model.classifier.dense = prune_linear_layer(model.classifier.dense, index, "
               "dim=1)\\n'\n"
               "        '            elif isinstance(model.classifier, torch.nn.Linear):\\n'\n"
               "        '                model.classifier = prune_linear_layer(model.classifier, index, dim=1)\\n'\n"
               '        \'        if hasattr(model, "cls"):\\n\'\n'
               '        \'            if hasattr(model.cls, "dense"):\\n\'\n'
               "        '                model.cls.dense = prune_linear_layer(model.classifier.dense, index, "
               "dim=1)\\n'\n"
               "        '        # MobileBertPooler: dense only exists when classifier_activation=True\\n'\n"
               '        \'        if hasattr(bert, "pooler") and bert.pooler is not None:\\n\'\n'
               '        \'            if hasattr(bert.pooler, "dense"):\\n\'\n'
               "        '                bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)\\n'\n"
               "        '                bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=0)\\n'\n"
               '        \'        if hasattr(model, "qa_outputs"):\\n\'\n'
               "        '            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)',\n"
               "        'cofi_utils: classifier/pooler — mobilebert + distilbert + bert')\n"
               '\n'
               '    # update_params model type detection\n'
               '    patch_file(utils_path,\n'
               "        'def update_params(model, zs):\\n'\n"
               '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\\n\'\n'
               "        '\\n'\n"
               "        '    config = model.config\\n'\n"
               "        '    hidden_dims = config.hidden_size\\n'\n"
               "        '    num_heads = config.num_attention_heads\\n'\n"
               "        '    dims_per_head = hidden_dims // num_heads\\n'\n"
               "        '    num_layers = config.num_hidden_layers',\n"
               "        'def update_params(model, zs):\\n'\n"
               '        \'    if hasattr(model, "mobilebert"):\\n\'\n'
               "        '        bert = model.mobilebert\\n'\n"
               "        '        num_layers = model.config.num_hidden_layers\\n'\n"
               "        '        is_mobilebert = True\\n'\n"
               '        \'    elif hasattr(model, "bert"):\\n\'\n'
               "        '        bert = model.bert\\n'\n"
               "        '        num_layers = model.config.num_hidden_layers\\n'\n"
               "        '        is_mobilebert = False\\n'\n"
               '        \'    elif hasattr(model, "roberta"):\\n\'\n'
               "        '        bert = model.roberta\\n'\n"
               "        '        num_layers = model.config.num_hidden_layers\\n'\n"
               "        '        is_mobilebert = False\\n'\n"
               '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
               "        '        bert = model.distilbert\\n'\n"
               "        '        num_layers = model.config.n_layers\\n'\n"
               "        '        is_mobilebert = False\\n'\n"
               "        '    else:\\n'\n"
               '        \'        raise ValueError(f"Unknown model type: {type(model)}")\\n\'\n'
               "        '\\n'\n"
               "        '    config = model.config\\n'\n"
               "        '    # MobileBERT: head masking operates at true_hidden_size (128), not hidden_size (512)\\n'\n"
               "        '    if is_mobilebert:\\n'\n"
               "        '        hidden_dims = config.true_hidden_size\\n'\n"
               "        '    else:\\n'\n"
               "        '        hidden_dims = config.hidden_size\\n'\n"
               "        '    num_heads = config.num_attention_heads\\n'\n"
               "        '    dims_per_head = hidden_dims // num_heads\\n'\n"
               "        '    num_layers = num_layers',\n"
               "        'cofi_utils: update_params model type detection + mobilebert true_hidden_size')\n"
               '\n'
               '    # update_params intermediate_z loop — mobilebert branch\n'
               '    patch_file(utils_path,\n'
               "        '            for layer in range(num_layers):\\n'\n"
               '        \'                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()\\n\'\n'
               "        '                bert.encoder.layer[layer].output.dense.weight.data = "
               "bert.encoder.layer[layer].output.dense.weight.data.mul(intermediate_z)\\n'\n"
               '        \'                if "mlp_z" in zs:\\n\'\n'
               '        \'                    mlp_z = zs["mlp_z"][layer].cpu()\\n\'\n'
               "        '                    bert.encoder.layer[layer].output.dense.weight.data = "
               "bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)\\n'\n"
               "        '                    bert.encoder.layer[layer].output.dense.bias.data = "
               "bert.encoder.layer[layer].output.dense.bias.data.mul(mlp_z)',\n"
               "        '            layers = _get_layers(bert)\\n'\n"
               "        '            for layer in range(num_layers):\\n'\n"
               '        \'                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()\\n\'\n'
               "        '                lyr = layers[layer]\\n'\n"
               "        '                if is_mobilebert:\\n'\n"
               "        '                    lyr.output.dense.weight.data = "
               "lyr.output.dense.weight.data.mul(intermediate_z)\\n'\n"
               '        \'                    if "mlp_z" in zs:\\n\'\n'
               '        \'                        mlp_z = zs["mlp_z"][layer].cpu()\\n\'\n'
               '        \'                        if hasattr(lyr.output, "bottleneck"):\\n\'\n'
               "        '                            lyr.output.bottleneck.dense.weight.data = "
               "lyr.output.bottleneck.dense.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)\\n'\n"
               "        '                            lyr.output.bottleneck.dense.bias.data = "
               "lyr.output.bottleneck.dense.bias.data.mul(mlp_z)\\n'\n"
               "        '                else:\\n'\n"
               "        '                    lyr.output.dense.weight.data = "
               "lyr.output.dense.weight.data.mul(intermediate_z)\\n'\n"
               '        \'                    if "mlp_z" in zs:\\n\'\n'
               '        \'                        mlp_z = zs["mlp_z"][layer].cpu()\\n\'\n'
               "        '                        lyr.output.dense.weight.data = "
               "lyr.output.dense.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)\\n'\n"
               "        '                        lyr.output.dense.bias.data = lyr.output.dense.bias.data.mul(mlp_z)',\n"
               "        'cofi_utils: update_params intermediate_z loop — mobilebert branch')\n"
               '\n'
               '    # update_params head_z loop — model-agnostic\n'
               '    patch_file(utils_path,\n'
               '        \'        if "head_z" in zs:\\n\'\n'
               "        '            for layer in range(num_layers):\\n'\n"
               '        \'                head_z = zs["head_z"][layer].cpu().squeeze().clone()\\n\'\n'
               "        '                head_z = torch.repeat_interleave(head_z, dims_per_head)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.value.weight.data = "
               'bert.encoder.layer[layer].attention.self.value.weight.transpose(0, 1).data.mul(head_z).transpose(0, '
               "1)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.value.bias.data = "
               "bert.encoder.layer[layer].attention.self.value.bias.data.mul(head_z)\\n'\n"
               '        \'                if "head_layer_z" in zs:\\n\'\n'
               '        \'                    head_layer_z = zs["head_layer_z"][layer].cpu()\\n\'\n'
               "        '                    bert.encoder.layer[layer].attention.output.dense.weight.data = "
               "bert.encoder.layer[\\n'\n"
               "        '                        layer].attention.output.dense.weight.transpose(0, "
               "1).data.mul(head_layer_z).transpose(0, 1)\\n'\n"
               "        '                    bert.encoder.layer[layer].attention.output.dense.bias.data = "
               "bert.encoder.layer[\\n'\n"
               "        '                        layer].attention.output.dense.bias.data.mul(head_layer_z)',\n"
               '        \'        if "head_z" in zs:\\n\'\n'
               "        '            layers = _get_layers(bert)\\n'\n"
               "        '            for layer in range(num_layers):\\n'\n"
               '        \'                head_z = zs["head_z"][layer].cpu().squeeze().clone()\\n\'\n'
               "        '                head_z = torch.repeat_interleave(head_z, dims_per_head)\\n'\n"
               "        '                lyr = layers[layer]\\n'\n"
               "        '                if is_mobilebert:\\n'\n"
               "        '                    v = lyr.attention.self.value\\n'\n"
               "        '                    o = lyr.attention.output.dense\\n'\n"
               "        '                else:\\n'\n"
               "        '                    v = lyr.attention.self.value\\n'\n"
               "        '                    o = lyr.attention.output.dense\\n'\n"
               "        '                if v is not None:\\n'\n"
               "        '                    v.weight.data = v.weight.transpose(0, 1).data.mul(head_z).transpose(0, "
               "1)\\n'\n"
               "        '                    v.bias.data = v.bias.data.mul(head_z)\\n'\n"
               '        \'                if "head_layer_z" in zs:\\n\'\n'
               '        \'                    head_layer_z = zs["head_layer_z"][layer].cpu()\\n\'\n'
               "        '                    if o is not None:\\n'\n"
               "        '                        o.weight.data = o.weight.transpose(0, "
               "1).data.mul(head_layer_z).transpose(0, 1)\\n'\n"
               "        '                        o.bias.data = o.bias.data.mul(head_layer_z)',\n"
               "        'cofi_utils: update_params head_z loop — model-agnostic')\n"
               '\n'
               '    # update_params hidden_z embeddings + layers — mobilebert branch\n'
               '    patch_file(utils_path,\n'
               '        \'            hidden_z = zs["hidden_z"].cpu().squeeze().clone()\\n\'\n'
               "        '            bert.embeddings.word_embeddings.weight.data =\\\\\\n'\n"
               "        '                bert.embeddings.word_embeddings.weight.data.mul(hidden_z)\\n'\n"
               "        '            bert.embeddings.position_embeddings.weight.data = \\\\\\n'\n"
               "        '                bert.embeddings.position_embeddings.weight.data.mul(hidden_z)\\n'\n"
               "        '            bert.embeddings.token_type_embeddings.weight.data = \\\\\\n'\n"
               "        '                bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)\\n'\n"
               "        '            for layer in range(num_layers):\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.key.weight.data = "
               "bert.encoder.layer[layer].attention.self.key.weight.data.mul(hidden_z)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.query.weight.data = "
               "bert.encoder.layer[layer].attention.self.query.weight.data.mul(hidden_z)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.self.value.weight.data = "
               "bert.encoder.layer[layer].attention.self.value.weight.data.mul(hidden_z)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.output.dense.weight.data = "
               'bert.encoder.layer[layer].attention.output.dense.weight.data.transpose(0, '
               "1).mul(hidden_z).transpose(0, 1)\\n'\n"
               "        '                bert.encoder.layer[layer].attention.output.dense.bias.data = "
               "bert.encoder.layer[layer].attention.output.dense.bias.data.mul(hidden_z)\\n'\n"
               "        '                bert.encoder.layer[layer].intermediate.dense.weight.data = "
               "bert.encoder.layer[layer].intermediate.dense.weight.data.mul(hidden_z)\\n'\n"
               "        '                bert.encoder.layer[layer].output.dense.weight.data = "
               "bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)\\n'\n"
               '        \'            if hasattr(bert.pooler, "dense"):\\n\'\n'
               "        '                bert.pooler.dense.weight.data = "
               "bert.pooler.dense.weight.data.mul(hidden_z)\\n'\n"
               '        \'            if hasattr(model, "qa_outputs"):\\n\'\n'
               "        '                model.qa_outputs.weight.data = model.qa_outputs.weight.data.mul(hidden_z)',\n"
               '        \'            hidden_z = zs["hidden_z"].cpu().squeeze().clone()\\n\'\n'
               "        '            # MobileBERT: word_embeddings is 128-dim (skip), position+token_type are "
               "512-dim\\n'\n"
               "        '            if not is_mobilebert:\\n'\n"
               "        '                bert.embeddings.word_embeddings.weight.data = \\\\\\n'\n"
               "        '                    bert.embeddings.word_embeddings.weight.data.mul(hidden_z)\\n'\n"
               "        '            bert.embeddings.position_embeddings.weight.data = \\\\\\n'\n"
               "        '                bert.embeddings.position_embeddings.weight.data.mul(hidden_z)\\n'\n"
               '        \'            if hasattr(bert.embeddings, "token_type_embeddings"):\\n\'\n'
               "        '                bert.embeddings.token_type_embeddings.weight.data = \\\\\\n'\n"
               "        '                    bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)\\n'\n"
               "        '            layers = _get_layers(bert)\\n'\n"
               "        '            for layer in range(num_layers):\\n'\n"
               "        '                lyr = layers[layer]\\n'\n"
               "        '                if is_mobilebert:\\n'\n"
               "        '                    # MobileBERT hidden_z (512-dim) only applies to OutputBottleneck\\n'\n"
               '        \'                    if hasattr(lyr.output, "bottleneck"):\\n\'\n'
               "        '                        bn = lyr.output.bottleneck\\n'\n"
               "        '                        bn.dense.weight.data = bn.dense.weight.data.transpose(0, "
               "1).mul(hidden_z).transpose(0, 1)\\n'\n"
               "        '                        bn.dense.bias.data = bn.dense.bias.data.mul(hidden_z)\\n'\n"
               "        '                else:\\n'\n"
               "        '                    lyr.attention.self.key.weight.data = "
               "lyr.attention.self.key.weight.data.mul(hidden_z)\\n'\n"
               "        '                    lyr.attention.self.query.weight.data = "
               "lyr.attention.self.query.weight.data.mul(hidden_z)\\n'\n"
               "        '                    lyr.attention.self.value.weight.data = "
               "lyr.attention.self.value.weight.data.mul(hidden_z)\\n'\n"
               "        '                    lyr.attention.output.dense.weight.data = "
               "lyr.attention.output.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)\\n'\n"
               "        '                    lyr.attention.output.dense.bias.data = "
               "lyr.attention.output.dense.bias.data.mul(hidden_z)\\n'\n"
               "        '                    lyr.intermediate.dense.weight.data = "
               "lyr.intermediate.dense.weight.data.mul(hidden_z)\\n'\n"
               "        '                    lyr.output.dense.weight.data = lyr.output.dense.weight.data.transpose(0, "
               "1).mul(hidden_z).transpose(0, 1)\\n'\n"
               '        \'            if hasattr(bert, "pooler") and bert.pooler is not None and hasattr(bert.pooler, '
               '"dense"):\\n\'\n'
               "        '                bert.pooler.dense.weight.data = "
               "bert.pooler.dense.weight.data.mul(hidden_z)\\n'\n"
               '        \'            if hasattr(model, "pre_classifier"):\\n\'\n'
               "        '                model.pre_classifier.weight.data = "
               "model.pre_classifier.weight.data.mul(hidden_z)\\n'\n"
               '        \'            if hasattr(model, "qa_outputs"):\\n\'\n'
               "        '                model.qa_outputs.weight.data = model.qa_outputs.weight.data.mul(hidden_z)',\n"
               "        'cofi_utils: update_params hidden_z loop — mobilebert branch')\n"
               '\n'
               '    # prune_intermediate_layers — mobilebert/bert/distilbert + extra FFN stacks\n'
               '    patch_file(utils_path,\n'
               "        'def prune_intermediate_layers(model, keep_dims):\\n'\n"
               '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\\n\'\n'
               "        '    device = model.device\\n'\n"
               "        '    for layer in keep_dims:\\n'\n"
               "        '        if len(keep_dims[layer]) == 0:\\n'\n"
               "        '            bert.encoder.layer[layer].intermediate.dense = None\\n'\n"
               "        '            bert.encoder.layer[layer].output.dense = None\\n'\n"
               "        '        else:\\n'\n"
               "        '            bert.encoder.layer[layer].intermediate.dense = "
               'prune_linear_layer(bert.encoder.layer[layer].intermediate.dense, '
               "index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)\\n'\n"
               "        '            bert.encoder.layer[layer].output.dense = "
               'prune_linear_layer(bert.encoder.layer[layer].output.dense, '
               "index=torch.LongTensor(keep_dims[layer]).to(device), dim=1)',\n"
               "        'def prune_intermediate_layers(model, keep_dims):\\n'\n"
               '        \'    if hasattr(model, "mobilebert"):\\n\'\n'
               "        '        bert = model.mobilebert\\n'\n"
               '        \'    elif hasattr(model, "bert"):\\n\'\n'
               "        '        bert = model.bert\\n'\n"
               '        \'    elif hasattr(model, "roberta"):\\n\'\n'
               "        '        bert = model.roberta\\n'\n"
               '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
               "        '        bert = model.distilbert\\n'\n"
               "        '    else:\\n'\n"
               '        \'        raise ValueError(f"Unknown model type: {type(model)}")\\n\'\n'
               "        '    device = model.device\\n'\n"
               '        \'    is_mobilebert = hasattr(model, "mobilebert")\\n\'\n'
               "        '    layers = _get_layers(bert)\\n'\n"
               "        '    for layer in keep_dims:\\n'\n"
               "        '        lyr = layers[layer]\\n'\n"
               "        '        idx = torch.LongTensor(keep_dims[layer]).to(device) if keep_dims[layer] else "
               "None\\n'\n"
               "        '        if is_mobilebert:\\n'\n"
               "        '            if len(keep_dims[layer]) == 0:\\n'\n"
               "        '                lyr.intermediate.dense = None\\n'\n"
               "        '                lyr.output.dense = None\\n'\n"
               "        '            else:\\n'\n"
               "        '                lyr.intermediate.dense = prune_linear_layer(lyr.intermediate.dense, "
               "index=idx, dim=0)\\n'\n"
               "        '                lyr.output.dense       = prune_linear_layer(lyr.output.dense,       "
               "index=idx, dim=1)\\n'\n"
               "        '            # Also structurally prune the extra FFN stacks (ffn[0..2])\\n'\n"
               '        \'            if hasattr(lyr, "ffn") and len(lyr.ffn) > 0:\\n\'\n'
               "        '                for j in range(len(lyr.ffn)):\\n'\n"
               "        '                    if len(keep_dims[layer]) == 0:\\n'\n"
               "        '                        lyr.ffn[j].intermediate.dense = None\\n'\n"
               "        '                        lyr.ffn[j].output.dense = None\\n'\n"
               "        '                    else:\\n'\n"
               "        '                        lyr.ffn[j].intermediate.dense = prune_linear_layer(\\n'\n"
               "        '                            lyr.ffn[j].intermediate.dense, index=idx, dim=0)\\n'\n"
               "        '                        lyr.ffn[j].output.dense = prune_linear_layer(\\n'\n"
               "        '                            lyr.ffn[j].output.dense, index=idx, dim=1)\\n'\n"
               '        \'        elif hasattr(lyr, "intermediate"):\\n\'\n'
               "        '            if len(keep_dims[layer]) == 0:\\n'\n"
               "        '                lyr.intermediate.dense = None\\n'\n"
               "        '                lyr.output.dense = None\\n'\n"
               "        '            else:\\n'\n"
               "        '                lyr.intermediate.dense = prune_linear_layer(lyr.intermediate.dense, "
               "index=idx, dim=0)\\n'\n"
               "        '                lyr.output.dense       = prune_linear_layer(lyr.output.dense,       "
               "index=idx, dim=1)\\n'\n"
               '        \'        elif hasattr(lyr, "ffn_lin1"):\\n\'\n'
               "        '            if len(keep_dims[layer]) == 0:\\n'\n"
               "        '                lyr.ffn_lin1 = lyr.ffn_lin2 = None\\n'\n"
               "        '            else:\\n'\n"
               "        '                lyr.ffn_lin1 = prune_linear_layer(lyr.ffn_lin1, index=idx, dim=0)\\n'\n"
               "        '                lyr.ffn_lin2 = prune_linear_layer(lyr.ffn_lin2, index=idx, dim=1)',\n"
               "        'cofi_utils: prune_intermediate_layers — mobilebert/bert/distilbert + extra FFN stacks')\n"
               '\n'
               '    # prune_model_with_z: bottleneck.attention zeroing for fully-pruned-heads layers\n'
               '    patch_file(utils_path,\n'
               "        '        model.prune_heads(prune_heads)',\n"
               "        '        model.prune_heads(prune_heads)\\n'\n"
               "        '\\n'\n"
               "        '        # MobileBERT: when ALL heads pruned, bottleneck.attention becomes dead weight\\n'\n"
               '        \'        if hasattr(model, "mobilebert"):\\n\'\n'
               "        '            bert_for_bn = model.mobilebert\\n'\n"
               "        '            layers_for_bn = _get_layers(bert_for_bn)\\n'\n"
               "        '            for layer, heads_idx in prune_heads.items():\\n'\n"
               "        '                n_heads_total = head_z[layer].numel()\\n'\n"
               "        '                if len(heads_idx) == n_heads_total:\\n'\n"
               "        '                    lyr_bn = layers_for_bn[layer]\\n'\n"
               '        \'                    if hasattr(lyr_bn, "bottleneck") and hasattr(lyr_bn.bottleneck, '
               '"attention"):\\n\'\n'
               "        '                        lyr_bn.bottleneck.attention = None',\n"
               "        'cofi_utils: zero bottleneck.attention when all heads pruned')\n"
               '\n'
               '    # print loop range fix (range(0,12) -> range(_num_layers))\n'
               '    patch_file(utils_path,\n'
               "        '    for layer in range(0, 12):\\n'\n"
               '        \'        print("Layer:", layer)\\n\'\n'
               "        '        if bert.encoder.layer[layer].attention.self.query is not None:\\n'\n"
               '        \'            print("query:", '
               "bert.encoder.layer[layer].attention.self.query.weight.shape)\\n'\n"
               '        \'            print("key:", bert.encoder.layer[layer].attention.self.key.weight.shape)\\n\'\n'
               "        '        else:\\n'\n"
               '        \'            print("query:", None)\\n\'\n'
               '        \'            print("key:", None)\\n\'\n'
               "        '        if bert.encoder.layer[layer].attention.self.value is not None:\\n'\n"
               '        \'            print("value:", '
               "bert.encoder.layer[layer].attention.self.value.weight.shape)\\n'\n"
               '        \'            print("output:", '
               "bert.encoder.layer[layer].attention.output.dense.weight.shape)\\n'\n"
               "        '        else:\\n'\n"
               '        \'            print("value:", None)\\n\'\n'
               '        \'            print("output:", None)\\n\'\n'
               "        '        if bert.encoder.layer[layer].intermediate.dense is not None:\\n'\n"
               '        \'            print("up:", bert.encoder.layer[layer].intermediate.dense.weight.shape)\\n\'\n'
               '        \'            print("down:", bert.encoder.layer[layer].output.dense.weight.shape)\\n\'\n'
               "        '        else:\\n'\n"
               '        \'            print("up", None)\\n\'\n'
               '        \'            print("down", None)\',\n'
               '        \'    _num_layers = getattr(model.config, "num_hidden_layers", 12)\\n\'\n'
               "        '    for layer in range(0, _num_layers):\\n'\n"
               '        \'        print("Layer:", layer)\\n\'\n'
               "        '        if bert.encoder.layer[layer].attention.self.query is not None:\\n'\n"
               '        \'            print("query:", '
               "bert.encoder.layer[layer].attention.self.query.weight.shape)\\n'\n"
               '        \'            print("key:", bert.encoder.layer[layer].attention.self.key.weight.shape)\\n\'\n'
               "        '        else:\\n'\n"
               '        \'            print("query:", None)\\n\'\n'
               '        \'            print("key:", None)\\n\'\n'
               "        '        if bert.encoder.layer[layer].attention.self.value is not None:\\n'\n"
               '        \'            print("value:", '
               "bert.encoder.layer[layer].attention.self.value.weight.shape)\\n'\n"
               '        \'            print("output:", '
               "bert.encoder.layer[layer].attention.output.dense.weight.shape)\\n'\n"
               "        '        else:\\n'\n"
               '        \'            print("value:", None)\\n\'\n'
               '        \'            print("output:", None)\\n\'\n'
               "        '        if bert.encoder.layer[layer].intermediate.dense is not None:\\n'\n"
               '        \'            print("up:", bert.encoder.layer[layer].intermediate.dense.weight.shape)\\n\'\n'
               '        \'            print("down:", bert.encoder.layer[layer].output.dense.weight.shape)\\n\'\n'
               "        '        else:\\n'\n"
               '        \'            print("up", None)\\n\'\n'
               '        \'            print("down", None)\',\n'
               "        'cofi_utils: print loop range — fix range(0,12) to num_hidden_layers')\n"
               '\n'
               '    # ── models/l0_module.py ────────────────────────────────────────────────────\n'
               "    print('\\n  Patching models/l0_module.py...')\n"
               "    l0_path = os.path.join(MOBILE_REPO_DIR, 'models', 'l0_module.py')\n"
               '\n'
               '    # config getattr fallbacks\n'
               '    patch_file(l0_path,\n'
               "        '        self.hidden_size = config.hidden_size\\n'\n"
               "        '        self.intermediate_size = config.intermediate_size \\n'\n"
               "        '        self.num_attention_heads = config.num_attention_heads\\n'\n"
               "        '        self.mlp_num_per_layer = 1\\n'\n"
               "        '        self.dim_per_head = self.hidden_size // self.num_attention_heads \\n'\n"
               "        '        self.num_hidden_layers = config.num_hidden_layers\\n'\n"
               "        '        self.vocab_size = config.vocab_size',\n"
               '        \'        self.hidden_size = getattr(config, "hidden_size", getattr(config, "dim", '
               "None))\\n'\n"
               '        \'        self.intermediate_size = getattr(config, "intermediate_size", getattr(config, '
               '"hidden_dim", None))\\n\'\n'
               '        \'        self.num_attention_heads = getattr(config, "num_attention_heads", getattr(config, '
               '"n_heads", None))\\n\'\n'
               "        '        self.mlp_num_per_layer = 1\\n'\n"
               "        '        # MobileBERT: head masking at true_hidden_size, not hidden_size\\n'\n"
               '        \'        _head_hidden = getattr(config, "true_hidden_size", self.hidden_size)\\n\'\n'
               "        '        self.dim_per_head = _head_hidden // self.num_attention_heads\\n'\n"
               '        \'        self.num_hidden_layers = getattr(config, "num_hidden_layers", getattr(config, '
               '"n_layers", None))\\n\'\n'
               "        '        self.vocab_size = config.vocab_size',\n"
               "        'l0_module: config getattr fallbacks + mobilebert true_hidden_size for dim_per_head')\n"
               '\n'
               '    # params_per_head_layer/mlp_layer — use true_hidden_size + num_feedforward_networks\n'
               '    patch_file(l0_path,\n'
               "        '        self.params_per_head_layer = self.hidden_size * self.hidden_size * 4 + "
               "self.hidden_size * 4\\n'\n"
               "        '        self.params_per_head =  self.params_per_head_layer // self.num_attention_heads\\n'\n"
               "        '\\n\\n'\n"
               "        '        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * 2 + "
               "self.hidden_size + self.hidden_size * 4\\n'\n"
               "        '        self.params_per_intermediate_dim = self.params_per_mlp_layer // "
               "self.intermediate_size',\n"
               "        '        # MobileBERT: attention/FFN operate at true_hidden_size (128), not hidden_size "
               "(512)\\n'\n"
               "        '        # Using wrong dimension here inflates prunable_model_size denominator, causing\\n'\n"
               "        '        # expected_sparsity to read too high and Lagrangian to stop pruning too early.\\n'\n"
               '        \'        _attn_dim = getattr(config, "true_hidden_size", self.hidden_size)\\n\'\n'
               "        '        self.params_per_head_layer = _attn_dim * _attn_dim * 4 + _attn_dim * 4\\n'\n"
               "        '        self.params_per_head =  self.params_per_head_layer // self.num_attention_heads\\n'\n"
               "        '        # MobileBERT: num_feedforward_networks FFN stacks per layer (default 1 for BERT)\\n'\n"
               '        \'        _num_ffn = getattr(config, "num_feedforward_networks", 1)\\n\'\n'
               "        '        self.params_per_mlp_layer = (_attn_dim * self.intermediate_size * 2 + _attn_dim + "
               "_attn_dim * 4) * _num_ffn\\n'\n"
               "        '        self.params_per_intermediate_dim = self.params_per_mlp_layer // "
               "(self.intermediate_size * _num_ffn)',\n"
               "        'l0_module: params_per_head_layer/mlp_layer — true_hidden_size + num_feedforward_networks')\n"
               '\n'
               '    # ── trainer/trainer.py ─────────────────────────────────────────────────────\n'
               "    print('\\n  Patching trainer/trainer.py...')\n"
               "    trainer_path = os.path.join(MOBILE_REPO_DIR, 'trainer', 'trainer.py')\n"
               '    patch_file(trainer_path,\n'
               "        '* (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)',\n"
               "        '',\n"
               "        'trainer: remove world_size')\n"
               '    patch_file(trainer_path,\n'
               "        '                if self.start_prune:\\n                    zs = "
               "self.l0_module.forward(training=True)',\n"
               "        '                if self.start_prune and self.l0_module is not None:\\n                    zs "
               "= self.l0_module.forward(training=True)',\n"
               "        'trainer: guard l0_module None')\n"
               '    patch_file(trainer_path,\n'
               '        "torch.save(self.l0_module, os.path.join(output_dir, \'l0_module.pt\'))",\n'
               '        "if self.l0_module is not None:\\n            torch.save(self.l0_module, '
               'os.path.join(output_dir, \'l0_module.pt\'))",\n'
               "        'trainer: guard l0_module save')\n"
               '    patch_file(trainer_path,\n'
               "        '                else:\\n                    specified_teacher_layers = [2, 5, 8, 11]',\n"
               "        '                else:\\n'\n"
               "        '                    n_teacher_layers = teacher_outputs[2].__len__() - 1\\n'\n"
               "        '                    if n_teacher_layers >= 12:\\n'\n"
               "        '                        specified_teacher_layers = [2, 5, 8, 11]\\n'\n"
               "        '                    else:\\n'\n"
               "        '                        step = max(1, n_teacher_layers // 4)\\n'\n"
               "        '                        specified_teacher_layers = [min(i * step, n_teacher_layers - 1) for i "
               "in range(1, 5)]\\n'\n"
               "        '                        specified_teacher_layers = sorted(set(specified_teacher_layers))\\n'\n"
               "        '                        while len(specified_teacher_layers) < 4:\\n'\n"
               "        '                            specified_teacher_layers.append(n_teacher_layers - 1)',\n"
               "        'trainer: dynamic teacher layer indices')\n"
               '\n'
               "    print('\\nAll patches done.')\n"
               '\n'
               '# ── Block 1: Download pre-tuned models ────────────────────────────────────────\n'
               'def block1(tasks):\n'
               "    header('BLOCK 1 — Download Pre-tuned MobileBERT Models')\n"
               '    from transformers import AutoModelForSequenceClassification, AutoTokenizer\n'
               '\n'
               '    for task in tasks:\n'
               "        hf_id = MOBILE_PRETRAINED_FT.get(task, '')\n"
               '        if not hf_id:\n'
               "            print(f'[SKIP] {task}: no HuggingFace model ID set in MOBILE_PRETRAINED_FT')\n"
               '            continue\n'
               '        out = ft_dir(task)\n'
               '        os.makedirs(out, exist_ok=True)\n'
               '        if model_saved(out):\n'
               "            print(f'[SKIP] {task}: already at {out}')\n"
               '            continue\n'
               "        print(f'Downloading {hf_id} -> {out} ...')\n"
               '        try:\n'
               '            model = AutoModelForSequenceClassification.from_pretrained(\n'
               '                hf_id, trust_remote_code=True)\n'
               '            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)\n'
               '        except Exception as e:\n'
               "            print(f'  [FAILED] {hf_id} -> {e}')\n"
               '            print(f\'  Update MOBILE_PRETRAINED_FT["{task}"] with the correct model ID and retry.\')\n'
               '            continue\n'
               '        model.save_pretrained(out)\n'
               '        tok.save_pretrained(out)\n'
               '        del model, tok\n'
               "        print(f'  Saved to {out}')\n"
               '\n'
               "    print('\\nBlock 1 done.')\n"
               '\n'
               '\n'
               '# ── Block 2: CoFi pruning ──────────────────────────────────────────────────────\n'
               'def block2(tasks):\n'
               "    header('BLOCK 2 — CoFi Pruning (MobileBERT)')\n"
               '\n'
               '    if MOBILE_REPO_DIR not in sys.path:\n'
               '        sys.path.insert(0, MOBILE_REPO_DIR)\n'
               '\n'
               "    env = {**os.environ, 'HF_DATASETS_TRUST_REMOTE_CODE': '1'}\n"
               '\n'
               '    for task in tasks:\n'
               '        ft  = ft_dir(task)\n'
               '        out = pr_dir(task)\n'
               '        cfg = task_cfg(task)\n'
               '        os.makedirs(out, exist_ok=True)\n'
               '\n'
               '        if not model_saved(ft):\n'
               "            print(f'[ERROR] {task}: fine-tuned model missing at {ft}. Run --block 1 first.')\n"
               '            continue\n'
               '\n'
               "        best = os.path.join(out, 'best')\n"
               '        if model_saved(best):\n'
               "            print(f'[SKIP] {task}: already pruned at {best}')\n"
               '            continue\n'
               '\n'
               "        print(f'\\nPruning mobilebert/{task} -> {out}')\n"
               "        log_file = os.path.join(out, 'pruning_log.txt')\n"
               "        print(f'Log:     {log_file}')\n"
               "        print(f'Monitor: tail -f {log_file}')\n"
               '\n'
               '        cmd = [\n'
               '            sys.executable,\n'
               "            os.path.join(MOBILE_REPO_DIR, 'run_glue_prune.py'),\n"
               "            '--model_name_or_path', ft,\n"
               "            '--task_name', task,\n"
               "            '--do_train', '--do_eval',\n"
               "            '--max_seq_length', '128',\n"
               "            '--per_device_train_batch_size', '32',\n"
               "            '--per_device_eval_batch_size', '32',\n"
               "            '--learning_rate', '2e-5',\n"
               "            '--reg_learning_rate', '0.01',\n"
               "            '--num_train_epochs', str(cfg['prune_epochs']),\n"
               "            '--output_dir', out,\n"
               "            '--save_steps', str(cfg['save_steps']),\n"
               "            '--save_total_limit', '2',\n"
               "            '--eval_steps', str(cfg['eval_steps']),\n"
               "            '--eval_strategy', 'steps',\n"
               "            '--seed', str(SEED),\n"
               "            '--pruning_type', 'structured_heads+structured_mlp+hidden+layer',\n"
               "            '--target_sparsity', str(SPARSITY),\n"
               "            '--sparsity_epsilon', '0.01',\n"
               "            '--freeze_embeddings',\n"
               "            '--do_distill', '--do_layer_distill',\n"
               "            '--distillation_path', ft,\n"
               "            '--distill_ce_loss_alpha', '0.1',\n"
               "            '--distill_loss_alpha', '0.9',\n"
               "            '--distill_temp', '2',\n"
               "            '--layer_distill_version', str(cfg['layer_distill_v']),\n"
               "            '--prepruning_finetune_epochs', str(cfg['prepruning']),\n"
               "            '--lagrangian_warmup_epochs', str(cfg['lag_warmup']),\n"
               "            '--scheduler_type', 'linear',\n"
               "            '--local_rank', '-1',\n"
               "            '--report_to', 'none',\n"
               '        ]\n'
               '\n'
               "        with open(log_file, 'w') as log:\n"
               '            proc = subprocess.Popen(\n'
               '                cmd,\n'
               '                stdout=subprocess.PIPE,\n'
               '                stderr=subprocess.STDOUT,\n'
               '                text=True,\n'
               '                cwd=MOBILE_REPO_DIR,\n'
               '                env=env,\n'
               '            )\n'
               '            for line in proc.stdout:\n'
               '                sys.stdout.write(line)\n'
               '                sys.stdout.flush()\n'
               '                log.write(line)\n'
               '                log.flush()\n'
               '            proc.wait()\n'
               '\n'
               '        if model_saved(best):\n'
               "            print(f'\\n[DONE] {task}: best model at {best}')\n"
               '        else:\n'
               "            print(f'\\n[WARNING] {task}: no best/ checkpoint. Check {log_file}')\n"
               '\n'
               "    print('\\nBlock 2 done.')\n"
               '\n'
               '\n'
               '# ── Block 3: Evaluation ────────────────────────────────────────────────────────\n'
               'def block3(tasks):\n'
               "    header('BLOCK 3 — Evaluation (MobileBERT)')\n"
               '\n'
               '    import torch\n'
               '    from torch.utils.data import DataLoader\n'
               '    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig\n'
               '    from datasets import load_dataset\n'
               '    import evaluate as hf_evaluate\n'
               '\n'
               '    if MOBILE_REPO_DIR not in sys.path:\n'
               '        sys.path.insert(0, MOBILE_REPO_DIR)\n'
               '\n'
               '    # Auto-compute MOBILEBERT_BASE_PARAMS\n'
               '    global MOBILEBERT_BASE_PARAMS\n'
               '    if MOBILEBERT_BASE_PARAMS is None:\n'
               '        try:\n'
               '            from utils.utils import calculate_parameters\n'
               '            cfg_tmp = AutoConfig.from_pretrained(\n'
               "                'google/mobilebert-uncased', num_labels=2, trust_remote_code=True)\n"
               '            m_tmp = AutoModelForSequenceClassification.from_config(cfg_tmp)\n'
               '            MOBILEBERT_BASE_PARAMS = calculate_parameters(m_tmp)\n'
               "            print(f'[INFO] MOBILEBERT_BASE_PARAMS = {MOBILEBERT_BASE_PARAMS:,}')\n"
               '            del m_tmp\n'
               '        except Exception as e:\n'
               "            print(f'[WARN] Could not auto-compute MOBILEBERT_BASE_PARAMS: {e}')\n"
               '            MOBILEBERT_BASE_PARAMS = 1\n'
               '\n'
               '    TASK_KEYS = {\n'
               "        'sst2': ('sentence',  None),\n"
               "        'qnli': ('question',  'sentence'),\n"
               "        'mnli': ('premise',   'hypothesis'),\n"
               "        'qqp':  ('question1', 'question2'),\n"
               "        'rte':  ('sentence1', 'sentence2'),\n"
               '    }\n'
               '\n'
               '    def run_evaluation(model_path, task, label, out_dir):\n'
               '        result_file = eval_path(out_dir)\n'
               '        if os.path.exists(result_file):\n'
               "            print(f'[SKIP] {label}/{task}: already evaluated.')\n"
               '            r = json.load(open(result_file))\n'
               "            for k, v in r.items(): print(f'  {k}: {v}')\n"
               '            return r\n'
               '\n'
               "        print(f'\\nEvaluating [{label}] on {task} ...')\n"
               "        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
               "        print(f'  Device: {device}')\n"
               '\n'
               '        tok_path = (model_path\n'
               "                    if os.path.exists(os.path.join(model_path, 'tokenizer_config.json'))\n"
               '                    else ft_dir(task))\n'
               '        tok = AutoTokenizer.from_pretrained(tok_path)\n'
               '\n'
               '        # [HIGH] Use exact prefix match — NOT substring — "unpruned" contains "pruned"\n'
               "        if label.lower().strip().startswith('mobilebert pruned'):\n"
               '            from models.modeling_mobilebert import CoFiMobileBertForSequenceClassification\n'
               '            from utils.cofi_utils import load_zs, load_model\n'
               '            zs = load_zs(model_path)\n'
               '            if zs is not None:\n'
               '                model = load_model(model_path, CoFiMobileBertForSequenceClassification, zs)\n'
               '            else:\n'
               '                model = CoFiMobileBertForSequenceClassification.from_pretrained(model_path)\n'
               '        else:\n'
               '            model = AutoModelForSequenceClassification.from_pretrained(model_path)\n'
               '\n'
               '        n_params = sum(p.numel() for p in model.parameters())\n'
               '        mem_mb   = n_params * 4 / 1e6\n'
               '        model    = model.to(device).eval()\n'
               '\n'
               "        split = 'validation_matched' if task == 'mnli' else 'validation'\n"
               "        ds    = load_dataset('glue', task, trust_remote_code=True)[split]\n"
               '        col_a, col_b = TASK_KEYS[task]\n'
               '        ds = ds.select(range(min(1000, len(ds))))\n'
               '\n'
               '        def tokenize(batch):\n'
               '            args = ((batch[col_a],) if col_b is None\n'
               '                    else (batch[col_a], batch[col_b]))\n'
               "            return tok(*args, padding='max_length', truncation=True, max_length=128)\n"
               '\n'
               '        ds = ds.map(tokenize, batched=True,\n'
               '                    remove_columns=[c for c in ds.column_names\n'
               "                                    if c not in ['label', 'labels', 'idx']])\n"
               "        ds.set_format('torch')\n"
               '        loader = DataLoader(ds, batch_size=32)\n'
               '\n'
               '        all_preds, all_labels = [], []\n'
               '        total_time, total_examples = 0.0, 0\n'
               '\n'
               '        # Warmup\n'
               '        with torch.no_grad():\n'
               '            for batch in loader:\n'
               '                inp = {k: v.to(device) for k, v in batch.items()\n'
               "                       if k in ['input_ids', 'attention_mask', 'token_type_ids']}\n"
               '                model(**inp)\n'
               '                break\n'
               '\n'
               '        with torch.no_grad():\n'
               '            for batch in loader:\n'
               "                labels = batch.pop('label', batch.pop('labels', None))\n"
               "                batch.pop('idx', None)\n"
               '                inp = {k: v.to(device) for k, v in batch.items()\n'
               "                       if k in ['input_ids', 'attention_mask', 'token_type_ids']}\n"
               '                if torch.cuda.is_available(): torch.cuda.synchronize()\n'
               '                t0  = time.perf_counter()\n'
               '                out = model(**inp)\n'
               '                if torch.cuda.is_available(): torch.cuda.synchronize()\n'
               '                total_time     += time.perf_counter() - t0\n'
               '                total_examples += out.logits.shape[0]\n'
               '                all_preds.extend(out.logits.argmax(-1).cpu().numpy().tolist())\n'
               '                if labels is not None:\n'
               '                    all_labels.extend(labels.cpu().numpy().tolist())\n'
               '\n'
               '        latency_ms = (total_time / total_examples) * 1000\n'
               '        throughput  = total_examples / total_time\n'
               '\n'
               "        if task == 'qqp':\n"
               "            score = hf_evaluate.load('glue', 'qqp').compute(\n"
               "                predictions=all_preds, references=all_labels)['f1']\n"
               "            metric_name = 'F1'\n"
               "        elif task == 'mnli':\n"
               "            score = hf_evaluate.load('glue', 'mnli').compute(\n"
               "                predictions=all_preds, references=all_labels)['accuracy']\n"
               "            metric_name = 'Accuracy'\n"
               '        else:\n'
               "            score = hf_evaluate.load('glue', task).compute(\n"
               "                predictions=all_preds, references=all_labels)['accuracy']\n"
               "            metric_name = 'Accuracy'\n"
               '\n'
               '        # Sparsity via calculate_parameters (non-embedding, consistent with repo)\n'
               '        try:\n'
               '            from utils.utils import calculate_parameters\n'
               '            n_non_emb    = calculate_parameters(model)\n'
               '            sparsity_pct = max(0.0, (1 - n_non_emb / MOBILEBERT_BASE_PARAMS) * 100)\n'
               '        except Exception:\n'
               '            sparsity_pct = 0.0\n'
               '\n'
               '        results = {\n'
               "            'label':          label,\n"
               "            'task':           task,\n"
               "            'n_params':       n_params,\n"
               "            'memory_mb':      round(mem_mb, 2),\n"
               "            'latency_ms':     round(latency_ms, 4),\n"
               "            'throughput_eps': round(throughput, 2),\n"
               "            'sparsity_pct':   round(sparsity_pct, 2),\n"
               '            metric_name:      round(score, 4),\n'
               '        }\n'
               '\n'
               '        W = 52\n'
               "        print('=' * W)\n"
               "        print(f'  {label} — {task.upper()}')\n"
               "        print('=' * W)\n"
               "        print(f'  {metric_name:<24}: {score:.4f}')\n"
               "        print(f'  Sparsity %             : {sparsity_pct:.1f}%')\n"
               "        print(f'  Parameters             : {n_params:,}')\n"
               "        print(f'  Memory MB              : {mem_mb:.1f}')\n"
               "        print(f'  Latency ms/example     : {latency_ms:.3f}')\n"
               "        print(f'  Throughput ex/sec      : {throughput:.1f}')\n"
               "        print('=' * W)\n"
               '\n'
               '        os.makedirs(out_dir, exist_ok=True)\n'
               "        with open(result_file, 'w') as f:\n"
               '            json.dump(results, f, indent=2)\n'
               "        print(f'  Saved: {result_file}')\n"
               '\n'
               '        del model\n'
               '        if torch.cuda.is_available(): torch.cuda.empty_cache()\n'
               '        return results\n'
               '\n'
               '    for task in tasks:\n'
               "        print(f'\\n--- {task.upper()} unpruned ---')\n"
               "        run_evaluation(ft_dir(task), task, 'MobileBERT unpruned', ft_dir(task))\n"
               '\n'
               "        print(f'\\n--- {task.upper()} pruned 60% ---')\n"
               "        best = os.path.join(pr_dir(task), 'best')\n"
               '        if not model_saved(best):\n'
               "            print(f'  Not found: {best}. Run --block 2 --task {task} first.')\n"
               '        else:\n'
               "            run_evaluation(best, task, 'MobileBERT pruned 60%', pr_dir(task))\n"
               '\n'
               "    print('\\nBlock 3 done.')\n"
               '\n'
               '\n'
               '# ── Block 4: Results table ─────────────────────────────────────────────────────\n'
               'def block4():\n'
               "    header('BLOCK 4 — Full Results Table (MobileBERT)')\n"
               '\n'
               '    metric_label = {\n'
               "        'sst2': 'Accuracy', 'qnli': 'Accuracy', 'mnli': 'Accuracy',\n"
               "        'qqp': 'F1', 'rte': 'Accuracy',\n"
               '    }\n'
               '\n'
               '    def load_result(path):\n'
               '        p = eval_path(path)\n'
               '        if not os.path.exists(p): return None\n'
               '        return json.load(open(p))\n'
               '\n'
               '    W = 97\n'
               "    print('=' * W)\n"
               "    print('  COFI PRUNING RESULTS — MobileBERT on GLUE (60% Sparsity)')\n"
               "    print('=' * W)\n"
               '    print(f"  {\'Task\':<6} {\'Model\':<26} {\'Score\':>8} {\'Mem MB\':>9} "\n'
               '          f"{\'Lat ms\':>9} {\'Tput ex/s\':>11} {\'Sparsity\':>10}")\n'
               "    print('-' * W)\n"
               '\n'
               '    for task in ALL_TASKS:\n'
               '        ml  = metric_label[task]\n'
               '        unp = load_result(ft_dir(task))\n'
               '        pru = load_result(pr_dir(task))\n'
               '\n'
               '        def fmt(r):\n'
               "            if r is None: return ['N/A'] * 5\n"
               "            score = r.get(ml, 'N/A')\n"
               '            return [\n'
               "                f'{score:.4f}' if isinstance(score, float) else str(score),\n"
               "                str(r.get('memory_mb', 'N/A')),\n"
               "                str(r.get('latency_ms', 'N/A')),\n"
               "                str(r.get('throughput_eps', 'N/A')),\n"
               '                f"{r.get(\'sparsity_pct\', \'N/A\')}%",\n'
               '            ]\n'
               '\n'
               '        u, p = fmt(unp), fmt(pru)\n'
               '        print(f"  {task:<6} {\'MobileBERT unpruned\':<26} {u[0]:>8} {u[1]:>9} "\n'
               '              f"{u[2]:>9} {u[3]:>11} {u[4]:>10}")\n'
               '        print(f"  {\'\':<6} {\'MobileBERT pruned 60%\':<26} {p[0]:>8} {p[1]:>9} "\n'
               '              f"{p[2]:>9} {p[3]:>11} {p[4]:>10}")\n'
               '        if unp and pru:\n'
               '            try:\n'
               "                speedup   = unp['latency_ms'] / pru['latency_ms']\n"
               '                retention = float(p[0]) / float(u[0]) * 100\n'
               '                print(f"  {\'\':<6}   speedup {speedup:.2f}x | score retention {retention:.1f}%")\n'
               '            except Exception:\n'
               '                pass\n'
               "        print('-' * W)\n"
               '\n'
               "    print('=' * W)\n"
               "    print('  Accuracy for SST-2/QNLI/MNLI/RTE; F1 for QQP')\n"
               "    print('  Sparsity: non-embedding params vs MobileBERT baseline (auto-computed)')\n"
               '\n'
               '\n'
               '# ── Status ─────────────────────────────────────────────────────────────────────\n'
               'def show_status():\n'
               "    header('STATUS — MobileBERT')\n"
               '    W = 62\n'
               '    print(f"  {\'\':18}" + \'  \'.join(f\'{t:<6}\' for t in ALL_TASKS))\n'
               "    print('-' * W)\n"
               '    for label, fn in [\n'
               "        ('FT downloaded',  ft_dir),\n"
               "        ('Pruned',         lambda t: os.path.join(pr_dir(t), 'best')),\n"
               "        ('Eval unpruned',  ft_dir),\n"
               "        ('Eval pruned',    pr_dir),\n"
               '    ]:\n'
               "        if 'Eval' in label:\n"
               "            row = [('v' if os.path.exists(eval_path(fn(t))) else '.') for t in ALL_TASKS]\n"
               '        else:\n'
               "            row = [('v' if model_saved(fn(t)) else '.') for t in ALL_TASKS]\n"
               '        print(f"  {label:<18}" + \'  \'.join(f\'{s:<6}\' for s in row))\n'
               "    print('=' * W)\n"
               "    print('  v = done   . = not yet')\n"
               '\n'
               '\n'
               '# ── Main ───────────────────────────────────────────────────────────────────────\n'
               "if __name__ == '__main__':\n"
               "    parser = argparse.ArgumentParser(description='CoFiPruning MobileBERT on GLUE')\n"
               "    parser.add_argument('--block',  type=int, choices=[0, 1, 2, 3, 4])\n"
               "    parser.add_argument('--task',   type=str, choices=ALL_TASKS)\n"
               "    parser.add_argument('--status', action='store_true')\n"
               '    args = parser.parse_args()\n'
               '\n'
               '    tasks = [args.task] if args.task else ALL_TASKS\n'
               '\n'
               '    if args.status:       show_status()\n'
               '    elif args.block == 0: block0()\n'
               '    elif args.block == 1: block1(tasks)\n'
               '    elif args.block == 2: block2(tasks)\n'
               '    elif args.block == 3: block3(tasks)\n'
               '    elif args.block == 4: block4()\n'
               '    else:                 parser.print_help()',
 'tinybert': '"""\n'
             'CoFiPruning — TinyBERT-4L on 5 GLUE Tasks\n'
             '===========================================\n'
             'Run from the directory containing this script (e.g. ~/Cofi_stuff).\n'
             "All paths are relative to this script's location.\n"
             '\n'
             'TinyBERT-4L architecture:\n'
             '  - 4 layers (vs 12 in BERT-base)\n'
             '  - hidden_size = 312 (vs 768)\n'
             '  - intermediate_size = 1200 (vs 3072)\n'
             '  - num_attention_heads = 12 (same)\n'
             '  - Same BERT naming conventions (query/key/value, BertIntermediate, etc.)\n'
             '  - HuggingFace general: huawei-noah/TinyBERT_General_4L_312D\n'
             '\n'
             'Block 0 does a FRESH clone every time (rm -rf + git clone), then applies\n'
             'all patches. Safe to re-run.\n'
             '\n'
             'Usage:\n'
             '  python cofi_tinybert.py --status\n'
             '  python cofi_tinybert.py --block 0\n'
             '  python cofi_tinybert.py --block 1\n'
             '  python cofi_tinybert.py --block 1 --task rte\n'
             '  python cofi_tinybert.py --block 2\n'
             '  python cofi_tinybert.py --block 2 --task rte\n'
             '  python cofi_tinybert.py --block 3\n'
             '  python cofi_tinybert.py --block 3 --task rte\n'
             '  python cofi_tinybert.py --block 4\n'
             '"""\n'
             '\n'
             'import argparse\n'
             'import json\n'
             'import os\n'
             'import re\n'
             'import subprocess\n'
             'import sys\n'
             'import time\n'
             '\n'
             '# ── Paths (relative to this script) ─────────────────────────────────────────\n'
             'BASE_DIR = os.path.dirname(os.path.abspath(__file__))\n'
             "TINY_REPO_DIR = os.path.join(BASE_DIR, 'CoFiPruning_tinybert')\n"
             '\n'
             '# ── Constants ──────────────────────────────────────────────────────────────────\n'
             'SPARSITY    = 0.6\n'
             'SEED        = 57\n'
             "ALL_TASKS   = ['sst2', 'qnli', 'mnli', 'qqp', 'rte']\n"
             "SMALL_TASKS = {'rte', 'mrpc', 'cola', 'stsb'}\n"
             '\n'
             '# TinyBERT-4L fine-tuned models from HuggingFace (huawei-noah org).\n'
             '# Naming pattern: TinyBERT_4L_312D_<TASK> (task-specific distilled versions).\n'
             '# If a name 404s, block1 will report it — swap in the correct name and re-run.\n'
             'TINY_PRETRAINED_FT = {\n'
             "    'sst2': 'Sayan01/TinyBert4_sst2',\n"
             "    'qnli': 'Sayan01/tiny-bert-qnli-distilled',\n"
             "    'mnli': 'Sayan01/tiny-bert-mnli-m-distilled',\n"
             "    'qqp' : 'Sayan01/tiny-bert-qqp-distilled',\n"
             "    'rte' : 'Sayan01/tiny-bert-rte-distilled',\n"
             '}\n'
             "# Fallback general model if a task-specific one isn't available\n"
             "GENERAL_TINYBERT = 'huawei-noah/TinyBERT_General_4L_312D'\n"
             '\n'
             '# TinyBERT-4L non-embedding params:\n'
             'TINYBERT_BASE_PARAMS = 4_568_736  # calculate_parameters() on unpruned TinyBERT-4L (non-embedding)\n'
             '\n'
             'TINYBERT_EMBEDDING_PARAMS = 9_683_856  # sum(model.bert.embeddings.parameters())\n'
             '\n'
             '# ── Helpers ────────────────────────────────────────────────────────────────────\n'
             'def ft_dir(task):\n'
             "    return os.path.join(BASE_DIR, f'ft_tinybert_{task}')\n"
             '\n'
             'def pr_dir(task):\n'
             "    return os.path.join(BASE_DIR, f'pr_tinybert_{task}_s{int(SPARSITY*100)}')\n"
             '\n'
             'def eval_path(d):\n'
             "    return os.path.join(d, 'eval_results.json')\n"
             '\n'
             'def model_saved(path):\n'
             "    return (os.path.exists(os.path.join(path, 'pytorch_model.bin')) or\n"
             "            os.path.exists(os.path.join(path, 'model.safetensors')))\n"
             '\n'
             'def task_cfg_tiny(task):\n'
             '    if task in SMALL_TASKS:\n'
             '        return dict(prune_epochs=100, eval_steps=50,  save_steps=50,\n'
             '                    prepruning=4,  lag_warmup=20, layer_distill_v=4,\n'
             "                    reg_lr='0.1')\n"
             '    return dict(prune_epochs=50,  eval_steps=200, save_steps=200,\n'
             '                prepruning=2,  lag_warmup=5,  layer_distill_v=4,\n'
             "                reg_lr='0.1')\n"
             '\n'
             'def header(msg):\n'
             '    W = 60\n'
             "    print('\\n' + '=' * W)\n"
             "    print(f'  {msg}')\n"
             "    print('=' * W)\n"
             '\n'
             'def patch_file(fpath, old, new, description):\n'
             '    txt = open(fpath).read()\n'
             '    if new and new in txt:\n'
             "        print(f'  [already patched] {description}')\n"
             '        return\n'
             '    if old not in txt:\n'
             "        print(f'  [not found]       {description}')\n"
             '        return\n'
             "    open(fpath, 'w').write(txt.replace(old, new))\n"
             "    print(f'  [patched]         {description}')\n"
             '\n'
             '\n'
             '# ── Block 0: Fresh clone + patch repo ──────────────────────────────────────────\n'
             'def block0():\n'
             "    header('BLOCK 0 — Fresh clone + patch CoFiPruning for TinyBERT')\n"
             '\n'
             '    # ── Fresh clone ────────────────────────────────────────────────────────────\n'
             "    print('\\n  Refreshing repo...')\n"
             '    if os.path.exists(TINY_REPO_DIR):\n'
             "        subprocess.run(['rm', '-rf', TINY_REPO_DIR], check=True)\n"
             "        print(f'  Removed old {TINY_REPO_DIR}')\n"
             '    subprocess.run(\n'
             "        ['git', 'clone', 'https://github.com/princeton-nlp/CoFiPruning.git', TINY_REPO_DIR],\n"
             '        check=True)\n'
             "    print(f'  Cloned fresh into {TINY_REPO_DIR}')\n"
             '\n'
             '    # ── Standard import patches ────────────────────────────────────────────────\n'
             "    print('\\n  Standard patches...')\n"
             '    IMPORT_PATCHES = [\n'
             "        ('from transformers.file_utils import hf_bucket_url, cached_path',\n"
             "         'from huggingface_hub import cached_download as cached_path'),\n"
             "        ('from transformers.file_utils import cached_path',\n"
             "         'from huggingface_hub import cached_download as cached_path'),\n"
             "        ('from transformers.file_utils import hf_bucket_url', ''),\n"
             "        ('from datasets import load_dataset, load_metric, DatasetDict',\n"
             "         'from datasets import load_dataset, DatasetDict\\nimport evaluate'),\n"
             '        (\'metric = load_metric("glue", data_args.task_name)\',\n'
             '         \'metric = evaluate.load("glue", data_args.task_name)\'),\n'
             '        (\'metric = load_metric("accuracy")\',\n'
             '         \'metric = evaluate.load("accuracy")\'),\n'
             "        ('from black import main', ''),\n"
             '    ]\n'
             '    for root, dirs, files in os.walk(TINY_REPO_DIR):\n'
             "        dirs[:] = [d for d in dirs if d != '.git']\n"
             '        for fname in files:\n'
             "            if not fname.endswith('.py'):\n"
             '                continue\n'
             '            fpath = os.path.join(root, fname)\n'
             '            rel   = os.path.relpath(fpath, TINY_REPO_DIR)\n'
             '            for old, new in IMPORT_PATCHES:\n'
             "                patch_file(fpath, old, new, f'{rel}')\n"
             '\n'
             '    # ── run_glue_prune.py ──────────────────────────────────────────────────────\n'
             "    print('\\n  Patching run_glue_prune.py...')\n"
             "    glue_path = os.path.join(TINY_REPO_DIR, 'run_glue_prune.py')\n"
             '    patch_file(glue_path,\n'
             '        \'load_dataset("glue", data_args.task_name)\',\n'
             '        \'load_dataset("glue", data_args.task_name, trust_remote_code=True)\',\n'
             "        'add trust_remote_code')\n"
             '    patch_file(glue_path,\n'
             '        \'"evaluation_strategy"\',\n'
             '        \'"eval_strategy"\',\n'
             "        'evaluation_strategy -> eval_strategy')\n"
             '\n'
             '    # Model selection — local TinyBERT finetune paths don\'t match "bert*"\n'
             '    # or "distilbert" by string, so add an explicit local-path check.\n'
             '    patch_file(glue_path,\n'
             "        '    Model = CoFiBertForSequenceClassification if model_args.model_name_or_path.startswith(\\n'\n"
             '        \'        "bert") else CoFiRobertaForSequenceClassification\',\n'
             '        \'    if "distilbert" in model_args.model_name_or_path.lower():\\n\'\n'
             "        '        from models.modeling_distilbert import CoFiDistilBertForSequenceClassification\\n'\n"
             "        '        Model = CoFiDistilBertForSequenceClassification\\n'\n"
             '        \'    elif (model_args.model_name_or_path.startswith("bert")\\n\'\n'
             "        '          or os.path.exists(os.path.join(model_args.model_name_or_path, "
             '"config.json"))):\\n\'\n'
             '        \'        # "bert*" model IDs, and any local finetune directory (TinyBERT etc.)\\n\'\n'
             "        '        Model = CoFiBertForSequenceClassification\\n'\n"
             "        '    else:\\n'\n"
             "        '        Model = CoFiRobertaForSequenceClassification',\n"
             "        'run_glue_prune.py: model selection — local-path-aware')\n"
             '\n'
             '    # ── modeling_bert.py: remove broken from_pretrained override ──────────────\n'
             "    print('\\n  Patching models/modeling_bert.py...')\n"
             "    bert_path = os.path.join(TINY_REPO_DIR, 'models/modeling_bert.py')\n"
             '    src = open(bert_path).read()\n'
             '    pat = re.compile(\n'
             "        r'[ \\t]*@classmethod\\s*\\n[ \\t]*def from_pretrained\\(cls.*?(?=\\n[ \\t]{0,4}(?:def |class "
             "|\\Z))',\n"
             '        re.DOTALL)\n'
             '    if re.search(pat, src):\n'
             "        open(bert_path, 'w').write(re.sub(pat, '', src))\n"
             "        print('  [patched]         modeling_bert.py: removed from_pretrained override')\n"
             '    else:\n'
             "        print('  [already patched] modeling_bert.py: no from_pretrained override found')\n"
             '\n'
             '    # ── cofi_utils.py ─────────────────────────────────────────────────────────\n'
             "    print('\\n  Patching utils/cofi_utils.py...')\n"
             "    utils_path = os.path.join(TINY_REPO_DIR, 'utils/cofi_utils.py')\n"
             '\n'
             '    # safetensors support\n'
             '    patch_file(utils_path,\n'
             '        \'    p = os.path.join(model_path, "pytorch_model.bin")\\n\'\n'
             '        \'    loaded_weights = torch.load(p, map_location="cpu")\',\n'
             '        \'    p_bin = os.path.join(model_path, "pytorch_model.bin")\\n\'\n'
             '        \'    p_safe = os.path.join(model_path, "model.safetensors")\\n\'\n'
             "        '    if os.path.exists(p_bin):\\n'\n"
             '        \'        loaded_weights = torch.load(p_bin, map_location="cpu")\\n\'\n'
             "        '    elif os.path.exists(p_safe):\\n'\n"
             "        '        from safetensors.torch import load_file\\n'\n"
             "        '        loaded_weights = load_file(p_safe)\\n'\n"
             "        '    else:\\n'\n"
             '        \'        raise FileNotFoundError(f"No model weights found in {model_path}")\',\n'
             "        'cofi_utils.py: safetensors support')\n"
             '\n'
             '    # _get_layers helper before prune_model_with_z\n'
             '    patch_file(utils_path,\n'
             "        'def prune_model_with_z(zs, model):',\n"
             "        'def _get_layers(bert):\\n'\n"
             '        \'    if hasattr(bert, "encoder"):\\n\'\n'
             "        '        return bert.encoder.layer\\n'\n"
             '        \'    elif hasattr(bert, "transformer"):\\n\'\n'
             "        '        return bert.transformer.layer\\n'\n"
             "        '    else:\\n'\n"
             '        \'        raise ValueError(f"Cannot find layers in {type(bert)}")\\n\\n\'\n'
             "        'def prune_model_with_z(zs, model):',\n"
             "        'cofi_utils.py: add _get_layers helper')\n"
             '\n'
             '    # prune_model_with_z — model type + layer count detection\n'
             '    patch_file(utils_path,\n'
             "        'def prune_model_with_z(zs, model):\\n'\n"
             "        '    if zs is None:\\n'\n"
             "        '        return None, None\\n'\n"
             '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\',\n'
             "        'def prune_model_with_z(zs, model):\\n'\n"
             "        '    if zs is None:\\n'\n"
             "        '        return None, None\\n'\n"
             '        \'    if hasattr(model, "bert"):\\n\'\n'
             "        '        bert = model.bert\\n'\n"
             "        '        num_layers = model.config.num_hidden_layers\\n'\n"
             '        \'    elif hasattr(model, "roberta"):\\n\'\n'
             "        '        bert = model.roberta\\n'\n"
             "        '        num_layers = model.config.num_hidden_layers\\n'\n"
             '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
             "        '        bert = model.distilbert\\n'\n"
             "        '        num_layers = model.config.n_layers\\n'\n"
             "        '    else:\\n'\n"
             '        \'        raise ValueError(f"Unknown model type: {type(model)}")\',\n'
             "        'cofi_utils.py: prune_model_with_z — model type detection')\n"
             '\n'
             '    # hidden_z loop — bert/distilbert (also fixes range(0,12) -> range(num_layers))\n'
             '    patch_file(utils_path,\n'
             "        '        for layer in range(0, 12):\\n'\n"
             "        '            if bert.encoder.layer[layer].attention.self.query is not None:\\n'\n"
             "        '                bert.encoder.layer[layer].attention.self.query = \\\\\\n'\n"
             "        '                    prune_layer(bert.encoder.layer[layer].attention.self.query , index, "
             "dim=1)\\n'\n"
             "        '                bert.encoder.layer[layer].attention.self.key = \\\\\\n'\n"
             "        '                    prune_layer(bert.encoder.layer[layer].attention.self.key , index, "
             "dim=1)\\n'\n"
             "        '            if bert.encoder.layer[layer].attention.self.value is not None:\\n'\n"
             "        '                bert.encoder.layer[layer].attention.self.value = \\\\\\n'\n"
             "        '                    prune_layer(bert.encoder.layer[layer].attention.self.value , index, "
             "dim=1)\\n'\n"
             "        '                bert.encoder.layer[layer].attention.output.dense = \\\\\\n'\n"
             "        '                    prune_layer(bert.encoder.layer[layer].attention.output.dense , index, "
             "dim=0)\\n'\n"
             "        '                prune_layer_norm(bert.encoder.layer[layer].attention.output.LayerNorm, "
             "index)\\n'\n"
             "        '            if bert.encoder.layer[layer].intermediate.dense is not None:\\n'\n"
             "        '                bert.encoder.layer[layer].intermediate.dense = \\\\\\n'\n"
             "        '                    prune_layer( bert.encoder.layer[layer].intermediate.dense, index, "
             "dim=1)\\n'\n"
             "        '                bert.encoder.layer[layer].output.dense = \\\\\\n'\n"
             "        '                    prune_layer( bert.encoder.layer[layer].output.dense, index, dim=0)\\n'\n"
             "        '                prune_layer_norm(bert.encoder.layer[layer].output.LayerNorm, index)',\n"
             "        '        layers = _get_layers(bert)\\n'\n"
             "        '        for layer in range(num_layers):\\n'\n"
             "        '            lyr = layers[layer]\\n'\n"
             '        \'            if hasattr(lyr, "attention") and hasattr(lyr.attention, "self"):\\n\'\n'
             "        '                if lyr.attention.self.query is not None:\\n'\n"
             "        '                    lyr.attention.self.query = prune_layer(lyr.attention.self.query, index, "
             "dim=1)\\n'\n"
             "        '                    lyr.attention.self.key = prune_layer(lyr.attention.self.key, index, "
             "dim=1)\\n'\n"
             "        '                if lyr.attention.self.value is not None:\\n'\n"
             "        '                    lyr.attention.self.value = prune_layer(lyr.attention.self.value, index, "
             "dim=1)\\n'\n"
             "        '                    lyr.attention.output.dense = prune_layer(lyr.attention.output.dense, index, "
             "dim=0)\\n'\n"
             "        '                    prune_layer_norm(lyr.attention.output.LayerNorm, index)\\n'\n"
             "        '                if lyr.intermediate.dense is not None:\\n'\n"
             "        '                    lyr.intermediate.dense = prune_layer(lyr.intermediate.dense, index, "
             "dim=1)\\n'\n"
             "        '                    lyr.output.dense = prune_layer(lyr.output.dense, index, dim=0)\\n'\n"
             "        '                    prune_layer_norm(lyr.output.LayerNorm, index)\\n'\n"
             '        \'            elif hasattr(lyr, "attention") and hasattr(lyr.attention, "q_lin"):\\n\'\n'
             "        '                if lyr.attention.q_lin is not None:\\n'\n"
             "        '                    lyr.attention.q_lin = prune_layer(lyr.attention.q_lin, index, dim=1)\\n'\n"
             "        '                    lyr.attention.k_lin = prune_layer(lyr.attention.k_lin, index, dim=1)\\n'\n"
             "        '                if lyr.attention.v_lin is not None:\\n'\n"
             "        '                    lyr.attention.v_lin = prune_layer(lyr.attention.v_lin, index, dim=1)\\n'\n"
             "        '                    lyr.attention.out_lin = prune_layer(lyr.attention.out_lin, index, "
             "dim=0)\\n'\n"
             "        '                    prune_layer_norm(lyr.attn_output.LayerNorm, index)\\n'\n"
             "        '                if lyr.ffn_lin1 is not None:\\n'\n"
             "        '                    lyr.ffn_lin1 = prune_layer(lyr.ffn_lin1, index, dim=1)\\n'\n"
             "        '                    lyr.ffn_lin2 = prune_layer(lyr.ffn_lin2, index, dim=0)\\n'\n"
             "        '                    prune_layer_norm(lyr.ffn_output.LayerNorm, index)',\n"
             "        'cofi_utils.py: hidden_z loop — bert/distilbert + fix range(0,12)')\n"
             '\n'
             '    # print loop — model-agnostic (also fixes range(0,12))\n'
             '    patch_file(utils_path,\n'
             "        '    for layer in range(0, 12):\\n'\n"
             '        \'        print("Layer:", layer)\\n\'\n'
             "        '        if bert.encoder.layer[layer].attention.self.query is not None:\\n'\n"
             '        \'            print("query:", bert.encoder.layer[layer].attention.self.query.weight.shape)\\n\'\n'
             '        \'            print("key:", bert.encoder.layer[layer].attention.self.key.weight.shape)\\n\'\n'
             "        '        else:\\n'\n"
             '        \'            print("query:", None)\\n\'\n'
             '        \'            print("key:", None)\\n\'\n'
             "        '        if bert.encoder.layer[layer].attention.self.value is not None:\\n'\n"
             '        \'            print("value:", bert.encoder.layer[layer].attention.self.value.weight.shape)\\n\'\n'
             '        \'            print("output:", '
             "bert.encoder.layer[layer].attention.output.dense.weight.shape)\\n'\n"
             "        '        else:\\n'\n"
             '        \'            print("value:", None)\\n\'\n'
             '        \'            print("output:", None)\\n\'\n'
             "        '        if bert.encoder.layer[layer].intermediate.dense is not None:\\n'\n"
             '        \'            print("up:", bert.encoder.layer[layer].intermediate.dense.weight.shape)\\n\'\n'
             '        \'            print("down:", bert.encoder.layer[layer].output.dense.weight.shape)\\n\'\n'
             "        '        else:\\n'\n"
             '        \'            print("up", None)\\n\'\n'
             '        \'            print("down", None)\',\n'
             "        '    layers = _get_layers(bert)\\n'\n"
             "        '    for layer in range(num_layers):\\n'\n"
             '        \'        print("Layer:", layer)\\n\'\n'
             "        '        lyr = layers[layer]\\n'\n"
             '        \'        if hasattr(lyr, "attention") and hasattr(lyr.attention, "self"):\\n\'\n'
             "        '            q = lyr.attention.self.query\\n'\n"
             "        '            v = lyr.attention.self.value\\n'\n"
             "        '            up = lyr.intermediate.dense\\n'\n"
             "        '            down = lyr.output.dense\\n'\n"
             "        '        else:\\n'\n"
             '        \'            q = getattr(lyr.attention, "q_lin", None)\\n\'\n'
             '        \'            v = getattr(lyr.attention, "v_lin", None)\\n\'\n'
             '        \'            up = getattr(lyr, "ffn_lin1", None)\\n\'\n'
             '        \'            down = getattr(lyr, "ffn_lin2", None)\\n\'\n'
             '        \'        print("query:", q.weight.shape if q is not None else None)\\n\'\n'
             '        \'        print("key:", q.weight.shape if q is not None else None)\\n\'\n'
             '        \'        print("value:", v.weight.shape if v is not None else None)\\n\'\n'
             '        \'        print("up:", up.weight.shape if up is not None else None)\\n\'\n'
             '        \'        print("down:", down.weight.shape if down is not None else None)\',\n'
             "        'cofi_utils.py: print loop — model-agnostic + fix range(0,12)')\n"
             '\n'
             '    # pre_classifier (distilbert) + classifier + safe pooler\n'
             '    patch_file(utils_path,\n'
             '        \'        if hasattr(model, "classifier"):\\n\'\n'
             '        \'            if hasattr(model.classifier, "dense"):\\n\'\n'
             "        '                model.classifier.dense = prune_linear_layer(model.classifier.dense, index, "
             "dim=1)\\n'\n"
             '        \'        if hasattr(model, "cls"):\\n\'\n'
             '        \'            if hasattr(model.cls, "dense"):\\n\'\n'
             "        '                model.cls.dense = prune_linear_layer(model.classifier.dense, index, dim=1)\\n'\n"
             '        \'        if hasattr(bert.pooler, "dense"):\\n\'\n'
             "        '            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)\\n'\n"
             '        \'        if hasattr(model, "qa_outputs"):\\n\'\n'
             "        '            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)',\n"
             '        \'        if hasattr(model, "pre_classifier"):\\n\'\n'
             "        '            model.pre_classifier = prune_linear_layer(model.pre_classifier, index, dim=1)\\n'\n"
             "        '            model.pre_classifier = prune_linear_layer(model.pre_classifier, index, dim=0)\\n'\n"
             '        \'        if hasattr(model, "classifier"):\\n\'\n'
             '        \'            if hasattr(model.classifier, "dense"):\\n\'\n'
             "        '                model.classifier.dense = prune_linear_layer(model.classifier.dense, index, "
             "dim=1)\\n'\n"
             "        '            elif isinstance(model.classifier, torch.nn.Linear):\\n'\n"
             "        '                model.classifier = prune_linear_layer(model.classifier, index, dim=1)\\n'\n"
             '        \'        if hasattr(model, "cls"):\\n\'\n'
             '        \'            if hasattr(model.cls, "dense"):\\n\'\n'
             "        '                model.cls.dense = prune_linear_layer(model.classifier.dense, index, dim=1)\\n'\n"
             '        \'        if hasattr(bert, "pooler") and bert.pooler is not None and hasattr(bert.pooler, '
             '"dense"):\\n\'\n'
             "        '            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)\\n'\n"
             "        '            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=0)\\n'\n"
             '        \'        if hasattr(model, "qa_outputs"):\\n\'\n'
             "        '            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)',\n"
             "        'cofi_utils.py: pre_classifier + classifier + safe pooler')\n"
             '\n'
             '    # hidden_z embeddings section — guard token_type_embeddings for distilbert\n'
             '    patch_file(utils_path,\n'
             "        '        bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
             "        '            bert.embeddings.token_type_embeddings.weight.index_select(1, "
             "index).clone().detach())\\n'\n"
             "        '        bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]',\n"
             '        \'        if hasattr(bert.embeddings, "token_type_embeddings"):\\n\'\n'
             "        '            bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(\\n'\n"
             "        '                bert.embeddings.token_type_embeddings.weight.index_select(1, "
             "index).clone().detach())\\n'\n"
             "        '            bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]',\n"
             "        'cofi_utils.py: guard token_type_embeddings in hidden_z prune')\n"
             '\n'
             '    # prune_intermediate_layers — model-agnostic\n'
             '    patch_file(utils_path,\n'
             "        'def prune_intermediate_layers(model, keep_dims):\\n'\n"
             '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\\n\'\n'
             "        '    device = model.device\\n'\n"
             "        '    for layer in keep_dims:\\n'\n"
             "        '        if len(keep_dims[layer]) == 0:\\n'\n"
             "        '            bert.encoder.layer[layer].intermediate.dense = None\\n'\n"
             "        '            bert.encoder.layer[layer].output.dense = None\\n'\n"
             "        '        else:\\n'\n"
             "        '            bert.encoder.layer[layer].intermediate.dense = "
             'prune_linear_layer(bert.encoder.layer[layer].intermediate.dense, '
             "index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)\\n'\n"
             "        '            bert.encoder.layer[layer].output.dense = "
             'prune_linear_layer(bert.encoder.layer[layer].output.dense, '
             "index=torch.LongTensor(keep_dims[layer]).to(device), dim=1)',\n"
             "        'def prune_intermediate_layers(model, keep_dims):\\n'\n"
             '        \'    if hasattr(model, "bert"):\\n\'\n'
             "        '        bert = model.bert\\n'\n"
             '        \'    elif hasattr(model, "roberta"):\\n\'\n'
             "        '        bert = model.roberta\\n'\n"
             '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
             "        '        bert = model.distilbert\\n'\n"
             "        '    else:\\n'\n"
             '        \'        raise ValueError(f"Unknown model type: {type(model)}")\\n\'\n'
             "        '    device = model.device\\n'\n"
             "        '    layers = _get_layers(bert)\\n'\n"
             "        '    for layer in keep_dims:\\n'\n"
             "        '        lyr = layers[layer]\\n'\n"
             '        \'        is_distil = hasattr(lyr, "ffn_lin1")\\n\'\n'
             "        '        if len(keep_dims[layer]) == 0:\\n'\n"
             "        '            if is_distil:\\n'\n"
             "        '                lyr.ffn_lin1 = None\\n'\n"
             "        '                lyr.ffn_lin2 = None\\n'\n"
             "        '            else:\\n'\n"
             "        '                lyr.intermediate.dense = None\\n'\n"
             "        '                lyr.output.dense = None\\n'\n"
             "        '        else:\\n'\n"
             "        '            idx = torch.LongTensor(keep_dims[layer]).to(device)\\n'\n"
             "        '            if is_distil:\\n'\n"
             "        '                lyr.ffn_lin1 = prune_linear_layer(lyr.ffn_lin1, index=idx, dim=0)\\n'\n"
             "        '                lyr.ffn_lin2 = prune_linear_layer(lyr.ffn_lin2, index=idx, dim=1)\\n'\n"
             "        '            else:\\n'\n"
             "        '                lyr.intermediate.dense = prune_linear_layer(lyr.intermediate.dense, index=idx, "
             "dim=0)\\n'\n"
             "        '                lyr.output.dense = prune_linear_layer(lyr.output.dense, index=idx, dim=1)',\n"
             "        'cofi_utils.py: prune_intermediate_layers — model-agnostic')\n"
             '\n'
             '    # update_params — full model-agnostic rewrite\n'
             '    patch_file(utils_path,\n'
             "        'def update_params(model, zs):\\n'\n"
             '        \'    bert = model.bert if hasattr(model, "bert") else model.roberta\\n\'\n'
             "        '\\n'\n"
             "        '    config = model.config\\n'\n"
             "        '    hidden_dims = config.hidden_size\\n'\n"
             "        '    num_heads = config.num_attention_heads\\n'\n"
             "        '    dims_per_head = hidden_dims // num_heads\\n'\n"
             "        '    num_layers = config.num_hidden_layers\\n'\n"
             "        '\\n'\n"
             "        '    if zs is not None:\\n'\n"
             '        \'        if "intermediate_z" in zs:\\n\'\n'
             "        '            for layer in range(num_layers):\\n'\n"
             '        \'                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()\\n\'\n'
             "        '                bert.encoder.layer[layer].output.dense.weight.data = "
             "bert.encoder.layer[layer].output.dense.weight.data.mul(intermediate_z)\\n'\n"
             '        \'                if "mlp_z" in zs:\\n\'\n'
             '        \'                    mlp_z = zs["mlp_z"][layer].cpu()\\n\'\n'
             "        '                    bert.encoder.layer[layer].output.dense.weight.data = "
             "bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)\\n'\n"
             "        '                    bert.encoder.layer[layer].output.dense.bias.data = "
             "bert.encoder.layer[layer].output.dense.bias.data.mul(mlp_z)\\n'\n"
             "        '\\n'\n"
             '        \'        if "head_z" in zs:\\n\'\n'
             "        '            for layer in range(num_layers):\\n'\n"
             '        \'                head_z = zs["head_z"][layer].cpu().squeeze().clone()\\n\'\n'
             "        '                head_z = torch.repeat_interleave(head_z, dims_per_head)\\n'\n"
             "        '                bert.encoder.layer[layer].attention.self.value.weight.data = "
             'bert.encoder.layer[layer].attention.self.value.weight.transpose(0, 1).data.mul(head_z).transpose(0, '
             "1)\\n'\n"
             "        '                bert.encoder.layer[layer].attention.self.value.bias.data = "
             "bert.encoder.layer[layer].attention.self.value.bias.data.mul(head_z)\\n'\n"
             '        \'                if "head_layer_z" in zs:\\n\'\n'
             '        \'                    head_layer_z = zs["head_layer_z"][layer].cpu()\\n\'\n'
             "        '                    bert.encoder.layer[layer].attention.output.dense.weight.data = "
             "bert.encoder.layer[\\n'\n"
             "        '                        layer].attention.output.dense.weight.transpose(0, "
             "1).data.mul(head_layer_z).transpose(0, 1)\\n'\n"
             "        '                    bert.encoder.layer[layer].attention.output.dense.bias.data = "
             "bert.encoder.layer[\\n'\n"
             "        '                        layer].attention.output.dense.bias.data.mul(head_layer_z)\\n'\n"
             "        '\\n'\n"
             '        \'        if "hidden_z" in zs:\\n\'\n'
             '        \'            hidden_z = zs["hidden_z"].cpu().squeeze().clone()\\n\'\n'
             "        '            bert.embeddings.word_embeddings.weight.data =\\\\\\n'\n"
             "        '                bert.embeddings.word_embeddings.weight.data.mul(hidden_z)\\n'\n"
             "        '            bert.embeddings.position_embeddings.weight.data = \\\\\\n'\n"
             "        '                bert.embeddings.position_embeddings.weight.data.mul(hidden_z)\\n'\n"
             "        '            bert.embeddings.token_type_embeddings.weight.data = \\\\\\n'\n"
             "        '                bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)\\n'\n"
             "        '            for layer in range(num_layers):\\n'\n"
             "        '                bert.encoder.layer[layer].attention.self.key.weight.data = "
             "bert.encoder.layer[layer].attention.self.key.weight.data.mul(hidden_z)\\n'\n"
             "        '                bert.encoder.layer[layer].attention.self.query.weight.data = "
             "bert.encoder.layer[layer].attention.self.query.weight.data.mul(hidden_z)\\n'\n"
             "        '                bert.encoder.layer[layer].attention.self.value.weight.data = "
             "bert.encoder.layer[layer].attention.self.value.weight.data.mul(hidden_z)\\n'\n"
             "        '                bert.encoder.layer[layer].attention.output.dense.weight.data = "
             'bert.encoder.layer[layer].attention.output.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, '
             "1)\\n'\n"
             "        '                bert.encoder.layer[layer].attention.output.dense.bias.data = "
             "bert.encoder.layer[layer].attention.output.dense.bias.data.mul(hidden_z)\\n'\n"
             "        '                bert.encoder.layer[layer].intermediate.dense.weight.data = "
             "bert.encoder.layer[layer].intermediate.dense.weight.data.mul(hidden_z)\\n'\n"
             "        '                bert.encoder.layer[layer].output.dense.weight.data = "
             "bert.encoder.layer[layer].output.dense.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)\\n'\n"
             '        \'            if hasattr(bert.pooler, "dense"):\\n\'\n'
             "        '                bert.pooler.dense.weight.data = "
             "bert.pooler.dense.weight.data.mul(hidden_z)\\n'\n"
             '        \'            if hasattr(model, "qa_outputs"):\\n\'\n'
             "        '                model.qa_outputs.weight.data = model.qa_outputs.weight.data.mul(hidden_z)',\n"
             "        'def update_params(model, zs):\\n'\n"
             '        \'    if hasattr(model, "bert"):\\n\'\n'
             "        '        bert = model.bert\\n'\n"
             "        '        num_layers = model.config.num_hidden_layers\\n'\n"
             "        '        is_distil = False\\n'\n"
             '        \'    elif hasattr(model, "roberta"):\\n\'\n'
             "        '        bert = model.roberta\\n'\n"
             "        '        num_layers = model.config.num_hidden_layers\\n'\n"
             "        '        is_distil = False\\n'\n"
             '        \'    elif hasattr(model, "distilbert"):\\n\'\n'
             "        '        bert = model.distilbert\\n'\n"
             "        '        num_layers = model.config.n_layers\\n'\n"
             "        '        is_distil = True\\n'\n"
             "        '    else:\\n'\n"
             '        \'        raise ValueError(f"Unknown model type: {type(model)}")\\n\'\n'
             "        '\\n'\n"
             "        '    config = model.config\\n'\n"
             "        '    if is_distil:\\n'\n"
             "        '        hidden_dims = config.dim\\n'\n"
             "        '        num_heads = config.n_heads\\n'\n"
             "        '    else:\\n'\n"
             "        '        hidden_dims = config.hidden_size\\n'\n"
             "        '        num_heads = config.num_attention_heads\\n'\n"
             "        '    dims_per_head = hidden_dims // num_heads\\n'\n"
             "        '    layers = _get_layers(bert)\\n'\n"
             "        '\\n'\n"
             "        '    if zs is not None:\\n'\n"
             '        \'        if "intermediate_z" in zs:\\n\'\n'
             "        '            for layer in range(num_layers):\\n'\n"
             '        \'                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()\\n\'\n'
             "        '                lyr = layers[layer]\\n'\n"
             "        '                down = lyr.ffn_lin2 if is_distil else lyr.output.dense\\n'\n"
             "        '                down.weight.data = down.weight.data.mul(intermediate_z)\\n'\n"
             '        \'                if "mlp_z" in zs:\\n\'\n'
             '        \'                    mlp_z = zs["mlp_z"][layer].cpu()\\n\'\n'
             "        '                    down.weight.data = down.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, "
             "1)\\n'\n"
             "        '                    down.bias.data = down.bias.data.mul(mlp_z)\\n'\n"
             "        '\\n'\n"
             '        \'        if "head_z" in zs:\\n\'\n'
             "        '            for layer in range(num_layers):\\n'\n"
             '        \'                head_z = zs["head_z"][layer].cpu().squeeze().clone()\\n\'\n'
             "        '                head_z = torch.repeat_interleave(head_z, dims_per_head)\\n'\n"
             "        '                lyr = layers[layer]\\n'\n"
             "        '                if is_distil:\\n'\n"
             "        '                    v, o = lyr.attention.v_lin, lyr.attention.out_lin\\n'\n"
             "        '                else:\\n'\n"
             "        '                    v, o = lyr.attention.self.value, lyr.attention.output.dense\\n'\n"
             "        '                v.weight.data = v.weight.transpose(0, 1).data.mul(head_z).transpose(0, 1)\\n'\n"
             "        '                v.bias.data = v.bias.data.mul(head_z)\\n'\n"
             '        \'                if "head_layer_z" in zs:\\n\'\n'
             '        \'                    head_layer_z = zs["head_layer_z"][layer].cpu()\\n\'\n'
             "        '                    o.weight.data = o.weight.transpose(0, "
             "1).data.mul(head_layer_z).transpose(0, 1)\\n'\n"
             "        '                    o.bias.data = o.bias.data.mul(head_layer_z)\\n'\n"
             "        '\\n'\n"
             '        \'        if "hidden_z" in zs:\\n\'\n'
             '        \'            hidden_z = zs["hidden_z"].cpu().squeeze().clone()\\n\'\n'
             "        '            bert.embeddings.word_embeddings.weight.data = \\\\\\n'\n"
             "        '                bert.embeddings.word_embeddings.weight.data.mul(hidden_z)\\n'\n"
             "        '            bert.embeddings.position_embeddings.weight.data = \\\\\\n'\n"
             "        '                bert.embeddings.position_embeddings.weight.data.mul(hidden_z)\\n'\n"
             '        \'            if hasattr(bert.embeddings, "token_type_embeddings"):\\n\'\n'
             "        '                bert.embeddings.token_type_embeddings.weight.data = \\\\\\n'\n"
             "        '                    bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)\\n'\n"
             "        '            for layer in range(num_layers):\\n'\n"
             "        '                lyr = layers[layer]\\n'\n"
             "        '                if is_distil:\\n'\n"
             "        '                    q, k, v, o = lyr.attention.q_lin, lyr.attention.k_lin, lyr.attention.v_lin, "
             "lyr.attention.out_lin\\n'\n"
             "        '                    up, down = lyr.ffn_lin1, lyr.ffn_lin2\\n'\n"
             "        '                else:\\n'\n"
             "        '                    q, k, v = lyr.attention.self.query, lyr.attention.self.key, "
             "lyr.attention.self.value\\n'\n"
             "        '                    o = lyr.attention.output.dense\\n'\n"
             "        '                    up, down = lyr.intermediate.dense, lyr.output.dense\\n'\n"
             "        '                k.weight.data = k.weight.data.mul(hidden_z)\\n'\n"
             "        '                q.weight.data = q.weight.data.mul(hidden_z)\\n'\n"
             "        '                v.weight.data = v.weight.data.mul(hidden_z)\\n'\n"
             "        '                o.weight.data = o.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, "
             "1)\\n'\n"
             "        '                o.bias.data = o.bias.data.mul(hidden_z)\\n'\n"
             "        '                up.weight.data = up.weight.data.mul(hidden_z)\\n'\n"
             "        '                down.weight.data = down.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, "
             "1)\\n'\n"
             '        \'            if hasattr(model, "pre_classifier"):\\n\'\n'
             "        '                model.pre_classifier.weight.data = "
             "model.pre_classifier.weight.data.mul(hidden_z)\\n'\n"
             '        \'            elif hasattr(bert, "pooler") and bert.pooler is not None and hasattr(bert.pooler, '
             '"dense"):\\n\'\n'
             "        '                bert.pooler.dense.weight.data = "
             "bert.pooler.dense.weight.data.mul(hidden_z)\\n'\n"
             '        \'            if hasattr(model, "qa_outputs"):\\n\'\n'
             "        '                model.qa_outputs.weight.data = model.qa_outputs.weight.data.mul(hidden_z)',\n"
             "        'cofi_utils.py: update_params — full model-agnostic rewrite')\n"
             '\n'
             '    # ── l0_module.py: config attr name normalization ──────────────────────────\n'
             "    print('\\n  Patching models/l0_module.py...')\n"
             "    l0_path = os.path.join(TINY_REPO_DIR, 'models/l0_module.py')\n"
             '    patch_file(l0_path,\n'
             "        '        self.hidden_size = config.hidden_size\\n'\n"
             "        '        self.intermediate_size = config.intermediate_size \\n'\n"
             "        '        self.num_attention_heads = config.num_attention_heads\\n'\n"
             "        '        self.mlp_num_per_layer = 1\\n'\n"
             "        '        self.dim_per_head = self.hidden_size // self.num_attention_heads \\n'\n"
             "        '        self.num_hidden_layers = config.num_hidden_layers\\n'\n"
             "        '        self.vocab_size = config.vocab_size',\n"
             '        \'        self.hidden_size = getattr(config, "hidden_size", getattr(config, "dim", None))\\n\'\n'
             '        \'        self.intermediate_size = getattr(config, "intermediate_size", getattr(config, '
             '"hidden_dim", None))\\n\'\n'
             '        \'        self.num_attention_heads = getattr(config, "num_attention_heads", getattr(config, '
             '"n_heads", None))\\n\'\n'
             "        '        self.mlp_num_per_layer = 1\\n'\n"
             "        '        self.dim_per_head = self.hidden_size // self.num_attention_heads\\n'\n"
             '        \'        self.num_hidden_layers = getattr(config, "num_hidden_layers", getattr(config, '
             '"n_layers", None))\\n\'\n'
             "        '        self.vocab_size = config.vocab_size',\n"
             "        'l0_module.py: config getattr fix')\n"
             '\n'
             '    # ── trainer/trainer.py ─────────────────────────────────────────────────────\n'
             "    print('\\n  Patching trainer/trainer.py...')\n"
             "    trainer_path = os.path.join(TINY_REPO_DIR, 'trainer/trainer.py')\n"
             '    patch_file(trainer_path,\n'
             "        '* (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)',\n"
             "        '',\n"
             "        'trainer.py: remove world_size')\n"
             '    patch_file(trainer_path,\n'
             "        '                if self.start_prune:\\n                    zs = "
             "self.l0_module.forward(training=True)',\n"
             "        '                if self.start_prune and self.l0_module is not None:\\n                    zs = "
             "self.l0_module.forward(training=True)',\n"
             "        'trainer.py: guard l0_module None')\n"
             '    patch_file(trainer_path,\n'
             '        "torch.save(self.l0_module, os.path.join(output_dir, \'l0_module.pt\'))",\n'
             '        "if self.l0_module is not None:\\n            torch.save(self.l0_module, '
             'os.path.join(output_dir, \'l0_module.pt\'))",\n'
             "        'trainer.py: guard l0_module save')\n"
             '\n'
             '    # Dynamic teacher layer indices for <12-layer teacher (4-layer TinyBERT)\n'
             '    patch_file(trainer_path,\n'
             "        '                else:\\n                    specified_teacher_layers = [2, 5, 8, 11]',\n"
             "        '                else:\\n'\n"
             "        '                    n_teacher_layers = teacher_outputs[2].__len__() - 1\\n'\n"
             "        '                    if n_teacher_layers >= 12:\\n'\n"
             "        '                        specified_teacher_layers = [2, 5, 8, 11]\\n'\n"
             "        '                    else:\\n'\n"
             "        '                        step = max(1, n_teacher_layers // 4)\\n'\n"
             "        '                        specified_teacher_layers = [min(i * step, n_teacher_layers - 1) for i "
             "in range(1, 5)]\\n'\n"
             "        '                        specified_teacher_layers = sorted(set(specified_teacher_layers))\\n'\n"
             "        '                        while len(specified_teacher_layers) < 4:\\n'\n"
             "        '                            specified_teacher_layers.append(n_teacher_layers - 1)',\n"
             "        'trainer.py: dynamic teacher layer indices for <12-layer teacher')\n"
             '\n'
             "    print('\\nAll patches done.')\n"
             "    print('NOTE: TinyBERT uses BERT architecture — CoFiBertForSequenceClassification handles it.')\n"
             '\n'
             '\n'
             '# ── Block 1: Download pre-tuned TinyBERT models ───────────────────────────────\n'
             'def block1(tasks):\n'
             "    header('BLOCK 1 — Download Pre-tuned TinyBERT Models')\n"
             '    from transformers import AutoModelForSequenceClassification, AutoTokenizer\n'
             '\n'
             '    for task in tasks:\n'
             "        hf_id = TINY_PRETRAINED_FT.get(task, '')\n"
             '        out = ft_dir(task)\n'
             '        os.makedirs(out, exist_ok=True)\n'
             '        if model_saved(out):\n'
             "            print(f'[SKIP] {task}: already at {out}')\n"
             '            continue\n'
             '\n'
             "        print(f'Downloading {hf_id} -> {out} ...')\n"
             '        try:\n'
             '            model = AutoModelForSequenceClassification.from_pretrained(\n'
             '                hf_id, trust_remote_code=True)\n'
             '            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)\n'
             '        except Exception as e:\n'
             "            print(f'  [FAILED] {hf_id} -> {e}')\n"
             "            print(f'  Try GENERAL_TINYBERT ({GENERAL_TINYBERT}) and finetune manually,')\n"
             "            print(f'  or find the correct task-specific model name on HuggingFace')\n"
             '            print(f\'  and update TINY_PRETRAINED_FT["{task}"] in this script.\')\n'
             '            continue\n'
             '\n'
             '        model.save_pretrained(out)\n'
             '        tok.save_pretrained(out)\n'
             '        del model, tok\n'
             "        print(f'  Saved.')\n"
             '\n'
             "    print('\\nBlock 1 done.')\n"
             '\n'
             '\n'
             '# ── Block 2: CoFi pruning ──────────────────────────────────────────────────────\n'
             'def block2(tasks):\n'
             "    header('BLOCK 2 — CoFi Pruning (TinyBERT)')\n"
             '\n'
             '    if TINY_REPO_DIR not in sys.path:\n'
             '        sys.path.insert(0, TINY_REPO_DIR)\n'
             '\n'
             "    env = {**os.environ, 'HF_DATASETS_TRUST_REMOTE_CODE': '1'}\n"
             '\n'
             '    for task in tasks:\n'
             '        ft  = ft_dir(task)\n'
             '        out = pr_dir(task)\n'
             '        cfg = task_cfg_tiny(task)\n'
             '        os.makedirs(out, exist_ok=True)\n'
             '\n'
             '        if not model_saved(ft):\n'
             "            print(f'[ERROR] {task}: fine-tuned model missing at {ft}. Run --block 1 first.')\n"
             '            continue\n'
             '\n'
             "        best = os.path.join(out, 'best')\n"
             '        if model_saved(best):\n'
             "            print(f'[SKIP] {task}: already pruned at {best}')\n"
             '            continue\n'
             '\n'
             "        print(f'\\nPruning tinybert/{task} -> {out}')\n"
             "        log_file = os.path.join(out, 'pruning_log.txt')\n"
             "        print(f'Log:     {log_file}')\n"
             "        print(f'Monitor: tail -f {log_file}')\n"
             "        print('Ctrl+C stops safely — resumes from last checkpoint on next run.\\n')\n"
             '\n'
             '        cmd = [\n'
             '            sys.executable,\n'
             "            os.path.join(TINY_REPO_DIR, 'run_glue_prune.py'),\n"
             "            '--model_name_or_path', ft,\n"
             "            '--task_name', task,\n"
             "            '--do_train', '--do_eval',\n"
             "            '--max_seq_length', '128',\n"
             "            '--per_device_train_batch_size', '32',\n"
             "            '--per_device_eval_batch_size', '32',\n"
             "            '--learning_rate', '2e-5',\n"
             "            '--reg_learning_rate', cfg['reg_lr'],\n"
             "            '--num_train_epochs', str(cfg['prune_epochs']),\n"
             "            '--output_dir', out,\n"
             "            '--save_steps', str(cfg['save_steps']),\n"
             "            '--save_total_limit', '2',\n"
             "            '--eval_steps', str(cfg['eval_steps']),\n"
             "            '--eval_strategy', 'steps',\n"
             "            '--seed', str(SEED),\n"
             "            '--pruning_type', 'structured_heads+structured_mlp+hidden+layer',\n"
             "            '--target_sparsity', str(SPARSITY),\n"
             "            '--sparsity_epsilon', '0.01',\n"
             "            '--freeze_embeddings',\n"
             "            '--do_distill', '--do_layer_distill',\n"
             "            '--distillation_path', ft,\n"
             "            '--distill_ce_loss_alpha', '0.1',\n"
             "            '--distill_loss_alpha', '0.9',\n"
             "            '--distill_temp', '2',\n"
             "            '--layer_distill_version', str(cfg['layer_distill_v']),\n"
             "            '--prepruning_finetune_epochs', str(cfg['prepruning']),\n"
             "            '--lagrangian_warmup_epochs', str(cfg['lag_warmup']),\n"
             "            '--scheduler_type', 'linear',\n"
             "            '--local_rank', '-1',\n"
             "            '--report_to', 'none',\n"
             '        ]\n'
             '\n'
             "        with open(log_file, 'w') as log:\n"
             '            proc = subprocess.Popen(\n'
             '                cmd,\n'
             '                stdout=subprocess.PIPE,\n'
             '                stderr=subprocess.STDOUT,\n'
             '                text=True,\n'
             '                cwd=TINY_REPO_DIR,\n'
             '                env=env,\n'
             '            )\n'
             '            for line in proc.stdout:\n'
             '                sys.stdout.write(line)\n'
             '                sys.stdout.flush()\n'
             '                log.write(line)\n'
             '                log.flush()\n'
             '            proc.wait()\n'
             '\n'
             '        if model_saved(best):\n'
             "            print(f'\\n[DONE] {task}: best model at {best}')\n"
             '        else:\n'
             "            print(f'\\n[WARNING] {task}: no best/ checkpoint. Check {log_file}')\n"
             '\n'
             "    print('\\nBlock 2 done.')\n"
             '\n'
             '\n'
             '# ── Block 3: Evaluation ────────────────────────────────────────────────────────\n'
             'def block3(tasks):\n'
             "    header('BLOCK 3 — Evaluation (TinyBERT)')\n"
             '\n'
             '    import torch\n'
             '    from torch.utils.data import DataLoader\n'
             '    from transformers import AutoModelForSequenceClassification, AutoTokenizer\n'
             '    from datasets import load_dataset\n'
             '    import evaluate as hf_evaluate\n'
             '\n'
             '    if TINY_REPO_DIR not in sys.path:\n'
             '        sys.path.insert(0, TINY_REPO_DIR)\n'
             '\n'
             '    def run_evaluation(model_path, task, label, out_dir):\n'
             '        result_file = eval_path(out_dir)\n'
             '        if os.path.exists(result_file):\n'
             "            print(f'[SKIP] {label}/{task}: already evaluated.')\n"
             '            r = json.load(open(result_file))\n'
             '            for k, v in r.items():\n'
             "                print(f'  {k}: {v}')\n"
             '            return r\n'
             '\n'
             "        print(f'\\nEvaluating [{label}] on {task} ...')\n"
             "        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
             "        print(f'  Device: {device}')\n"
             '\n'
             '        tok_path = (ft_dir(task)\n'
             "                    if not os.path.exists(os.path.join(model_path, 'tokenizer_config.json'))\n"
             '                    else model_path)\n'
             '        tok = AutoTokenizer.from_pretrained(tok_path)\n'
             '\n'
             '        # \'pruned\' substring check is unsafe ("unpruned" contains "pruned") —\n'
             '        # match on the explicit "pruned 60%" label instead.\n'
             "        if label.lower().strip().startswith('tinybert pruned'):\n"
             '            from models.modeling_bert import CoFiBertForSequenceClassification\n'
             '            from utils.cofi_utils import load_zs, load_model\n'
             '            zs = load_zs(model_path)\n'
             '            if zs is None:\n'
             '                model = CoFiBertForSequenceClassification.from_pretrained(model_path)\n'
             '            else:\n'
             '                model = load_model(model_path, CoFiBertForSequenceClassification, zs)\n'
             '        else:\n'
             '            model = AutoModelForSequenceClassification.from_pretrained(model_path)\n'
             '\n'
             '        n_params = sum(p.numel() for p in model.parameters())\n'
             '        mem_mb   = n_params * 4 / 1e6\n'
             '        model    = model.to(device).eval()\n'
             '\n'
             "        if task == 'mnli':\n"
             "            ds = load_dataset('glue', 'mnli',\n"
             "                              trust_remote_code=True)['validation_matched']\n"
             '        else:\n'
             "            ds = load_dataset('glue', task,\n"
             "                              trust_remote_code=True)['validation']\n"
             '\n'
             '        col_map = {\n'
             "            'sst2': ('sentence',  None),\n"
             "            'qnli': ('question',  'sentence'),\n"
             "            'mnli': ('premise',   'hypothesis'),\n"
             "            'qqp' : ('question1', 'question2'),\n"
             "            'rte' : ('sentence1', 'sentence2'),\n"
             '        }\n'
             '        col_a, col_b = col_map[task]\n'
             '        ds = ds.select(range(min(1000, len(ds))))\n'
             '\n'
             '        def tokenize(batch):\n'
             '            args = ((batch[col_a],) if col_b is None\n'
             '                    else (batch[col_a], batch[col_b]))\n'
             "            return tok(*args, padding='max_length', truncation=True,\n"
             '                       max_length=128, return_tensors=None)\n'
             '\n'
             '        ds = ds.map(tokenize, batched=True,\n'
             '                    remove_columns=[c for c in ds.column_names\n'
             "                                    if c not in ['label', 'labels', 'idx']])\n"
             "        ds.set_format('torch')\n"
             '        loader = DataLoader(ds, batch_size=32)\n'
             '\n'
             '        all_preds, all_labels = [], []\n'
             '        total_time, total_examples = 0.0, 0\n'
             '\n'
             '        # warmup\n'
             '        with torch.no_grad():\n'
             '            for batch in loader:\n'
             '                inp = {k: v.to(device) for k, v in batch.items()\n'
             "                       if k in ['input_ids', 'attention_mask', 'token_type_ids']}\n"
             '                model(**inp)\n'
             '                break\n'
             '\n'
             '        with torch.no_grad():\n'
             '            for batch in loader:\n'
             "                labels = batch.pop('label', batch.pop('labels', None))\n"
             "                batch.pop('idx', None)\n"
             '                inp = {k: v.to(device) for k, v in batch.items()\n'
             "                       if k in ['input_ids', 'attention_mask', 'token_type_ids']}\n"
             '                if torch.cuda.is_available():\n'
             '                    torch.cuda.synchronize()\n'
             '                t0 = time.perf_counter()\n'
             '                out = model(**inp)\n'
             '                if torch.cuda.is_available():\n'
             '                    torch.cuda.synchronize()\n'
             '                t1 = time.perf_counter()\n'
             '                total_time     += (t1 - t0)\n'
             '                total_examples += out.logits.shape[0]\n'
             '                all_preds.extend(out.logits.argmax(-1).cpu().numpy().tolist())\n'
             '                if labels is not None:\n'
             '                    all_labels.extend(labels.cpu().numpy().tolist())\n'
             '\n'
             '        latency_ms = (total_time / total_examples) * 1000\n'
             '        throughput  = total_examples / total_time\n'
             '\n'
             "        if task == 'qqp':\n"
             "            score = hf_evaluate.load('glue', 'qqp').compute(\n"
             "                predictions=all_preds, references=all_labels)['f1']\n"
             "            metric_name = 'F1'\n"
             "        elif task == 'mnli':\n"
             "            score = hf_evaluate.load('glue', 'mnli').compute(\n"
             "                predictions=all_preds, references=all_labels)['accuracy']\n"
             "            metric_name = 'Accuracy'\n"
             '        else:\n'
             "            score = hf_evaluate.load('glue', task).compute(\n"
             "                predictions=all_preds, references=all_labels)['accuracy']\n"
             "            metric_name = 'Accuracy'\n"
             '\n'
             '        n_non_emb    = n_params - TINYBERT_EMBEDDING_PARAMS\n'
             '        sparsity_pct = max(0.0, (1 - n_non_emb / TINYBERT_BASE_PARAMS) * 100)\n'
             '\n'
             '        results = {\n'
             "            'label':          label,\n"
             "            'task':           task,\n"
             "            'n_params':       n_params,\n"
             "            'memory_mb':      round(mem_mb, 2),\n"
             "            'latency_ms':     round(latency_ms, 4),\n"
             "            'throughput_eps': round(throughput, 2),\n"
             "            'sparsity_pct':   round(sparsity_pct, 2),\n"
             '            metric_name:      round(score, 4),\n'
             '        }\n'
             '\n'
             '        W = 50\n'
             "        print('=' * W)\n"
             "        print(f'  {label} — {task.upper()}')\n"
             "        print('=' * W)\n"
             "        print(f'  {metric_name:<22}: {score:.4f}')\n"
             "        print(f'  Memory (MB)          : {mem_mb:.1f}')\n"
             "        print(f'  Latency (ms/example) : {latency_ms:.3f}')\n"
             "        print(f'  Throughput (ex/sec)  : {throughput:.1f}')\n"
             "        print(f'  Sparsity %           : {sparsity_pct:.1f}%')\n"
             "        print(f'  Parameters           : {n_params:,}')\n"
             "        print('=' * W)\n"
             '\n'
             '        os.makedirs(out_dir, exist_ok=True)\n'
             "        with open(result_file, 'w') as f:\n"
             '            json.dump(results, f, indent=2)\n'
             "        print(f'  Saved to {result_file}')\n"
             '\n'
             '        del model\n'
             '        if torch.cuda.is_available():\n'
             '            torch.cuda.empty_cache()\n'
             '        return results\n'
             '\n'
             '    for task in tasks:\n'
             "        print(f'\\n--- {task.upper()} unpruned ---')\n"
             "        run_evaluation(ft_dir(task), task, 'TinyBERT unpruned', ft_dir(task))\n"
             '\n'
             "        print(f'\\n--- {task.upper()} pruned (60% sparsity) ---')\n"
             "        best = os.path.join(pr_dir(task), 'best')\n"
             '        if not model_saved(best):\n'
             "            print(f'  Pruned model not found at {best}. Run --block 2 --task {task} first.')\n"
             '        else:\n'
             "            run_evaluation(best, task, 'TinyBERT pruned 60%', pr_dir(task))\n"
             '\n'
             "    print('\\nBlock 3 done.')\n"
             '\n'
             '\n'
             '# ── Block 4: Results summary ───────────────────────────────────────────────────\n'
             'def block4():\n'
             "    header('BLOCK 4 — Full Results Summary (TinyBERT)')\n"
             '\n'
             '    metric_label = {\n'
             "        'sst2': 'Accuracy', 'qnli': 'Accuracy', 'mnli': 'Accuracy',\n"
             "        'qqp': 'F1', 'rte': 'Accuracy',\n"
             '    }\n'
             '\n'
             '    def load_result(path):\n'
             '        p = eval_path(path)\n'
             '        if not os.path.exists(p):\n'
             '            return None\n'
             '        return json.load(open(p))\n'
             '\n'
             '    W = 97\n'
             "    print('=' * W)\n"
             "    print('  COFI PRUNING RESULTS — TinyBERT-4L on GLUE (60% Sparsity Target)')\n"
             "    print('=' * W)\n"
             '    print(f"  {\'Task\':<6} {\'Model\':<26} {\'Score\':>8} {\'Mem MB\':>9} "\n'
             '          f"{\'Lat ms\':>9} {\'Tput ex/s\':>11} {\'Sparsity\':>10}")\n'
             "    print('-' * W)\n"
             '\n'
             '    for task in ALL_TASKS:\n'
             '        ml  = metric_label[task]\n'
             '        unp = load_result(ft_dir(task))\n'
             '        pru = load_result(pr_dir(task))\n'
             '\n'
             '        def fmt(r):\n'
             '            if r is None:\n'
             "                return ['N/A'] * 5\n"
             "            score = r.get(ml, 'N/A')\n"
             '            return [\n'
             "                f'{score:.4f}' if isinstance(score, float) else str(score),\n"
             "                str(r.get('memory_mb', 'N/A')),\n"
             "                str(r.get('latency_ms', 'N/A')),\n"
             "                str(r.get('throughput_eps', 'N/A')),\n"
             '                f"{r.get(\'sparsity_pct\', \'N/A\')}%",\n'
             '            ]\n'
             '\n'
             '        u = fmt(unp)\n'
             '        p = fmt(pru)\n'
             '        print(f"  {task:<6} {\'TinyBERT unpruned\':<26} {u[0]:>8} {u[1]:>9} "\n'
             '              f"{u[2]:>9} {u[3]:>11} {u[4]:>10}")\n'
             '        print(f"  {\'\':<6} {\'TinyBERT pruned 60%\':<26} {p[0]:>8} {p[1]:>9} "\n'
             '              f"{p[2]:>9} {p[3]:>11} {p[4]:>10}")\n'
             '        if unp and pru:\n'
             '            try:\n'
             "                speedup   = unp['latency_ms'] / pru['latency_ms']\n"
             '                retention = float(p[0]) / float(u[0]) * 100\n'
             '                print(f"  {\'\':<6}   -> speedup {speedup:.2f}x   "\n'
             '                      f"score retention {retention:.1f}%")\n'
             '            except Exception:\n'
             '                pass\n'
             "        print('-' * W)\n"
             '\n'
             "    print('=' * W)\n"
             "    print('  Score: Accuracy for SST-2/QNLI/MNLI/RTE, F1 for QQP')\n"
             "    print('  Latency/Throughput: GPU, batch=32, 1000 validation examples')\n"
             "    print('  Sparsity: % reduction vs TinyBERT-4L non-embedding params (~14.35M)')\n"
             '\n'
             '\n'
             '# ── Status ─────────────────────────────────────────────────────────────────────\n'
             'def show_status():\n'
             "    header('STATUS — TinyBERT')\n"
             '    W = 62\n'
             '    print(f"  {\'\':18}" + \'  \'.join(f\'{t:<6}\' for t in ALL_TASKS))\n'
             "    print('-' * W)\n"
             '    rows = [\n'
             "        ('FT downloaded',  [('v' if model_saved(ft_dir(t)) else '.') for t in ALL_TASKS]),\n"
             "        ('Pruned',         [('v' if model_saved(os.path.join(pr_dir(t), 'best')) else '.') for t in "
             'ALL_TASKS]),\n'
             "        ('Eval unpruned',  [('v' if os.path.exists(eval_path(ft_dir(t))) else '.') for t in "
             'ALL_TASKS]),\n'
             "        ('Eval pruned',    [('v' if os.path.exists(eval_path(pr_dir(t))) else '.') for t in "
             'ALL_TASKS]),\n'
             '    ]\n'
             '    for label, row in rows:\n'
             '        print(f"  {label:<18}" + \'  \'.join(f\'{s:<6}\' for s in row))\n'
             "    print('=' * W)\n"
             "    print('  v = done   . = not yet')\n"
             '\n'
             '\n'
             '# ── Main ───────────────────────────────────────────────────────────────────────\n'
             "if __name__ == '__main__':\n"
             "    parser = argparse.ArgumentParser(description='CoFiPruning TinyBERT-4L on GLUE')\n"
             "    parser.add_argument('--block',  type=int, choices=[0, 1, 2, 3, 4])\n"
             "    parser.add_argument('--task',   type=str, choices=ALL_TASKS)\n"
             "    parser.add_argument('--status', action='store_true')\n"
             '    args = parser.parse_args()\n'
             '\n'
             '    tasks = [args.task] if args.task else ALL_TASKS\n'
             '\n'
             '    if args.status:\n'
             '        show_status()\n'
             '    elif args.block == 0:\n'
             '        block0()\n'
             '    elif args.block == 1:\n'
             '        block1(tasks)\n'
             '    elif args.block == 2:\n'
             '        block2(tasks)\n'
             '    elif args.block == 3:\n'
             '        block3(tasks)\n'
             '    elif args.block == 4:\n'
             '        block4()\n'
             '    else:\n'
             '        parser.print_help()'}


def build_model_aliases():
    aliases = {}
    for canonical, spec in MODEL_SPECS.items():
        aliases[canonical] = canonical
        for alias in spec["aliases"]:
            aliases[alias] = canonical
            aliases[alias.replace("_", "-")] = canonical
    return aliases


MODEL_ALIASES = build_model_aliases()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a model-specific CoFi pruning workflow."
    )
    parser.add_argument(
        "--model",
        help=(
            "Model to run. Choices: "
            + ", ".join(MODEL_SPECS)
            + ". Aliases like bertbase/base/bert-base are accepted."
        ),
    )
    parser.add_argument("--block", type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--task", type=str)
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported model names and aliases, then exit.",
    )
    return parser, parser.parse_args(argv)


def print_models():
    print("Supported models:")
    for canonical, spec in MODEL_SPECS.items():
        aliases = ", ".join(spec["aliases"])
        print(f"  {canonical:<10} {spec['label']:<12} aliases: {aliases}")


def resolve_model(name, parser):
    key = name.lower().replace("_", "-")
    canonical = MODEL_ALIASES.get(key)
    if canonical is None:
        parser.error(
            f"unknown --model {name!r}; use one of: {', '.join(MODEL_SPECS)}"
        )
    return canonical


def load_model_namespace(canonical):
    namespace = {
        "__name__": f"_embedded_{canonical}",
        "__file__": __file__,
        "__package__": None,
    }
    exec(compile(MODEL_SOURCES[canonical], f"<embedded {canonical}>", "exec"), namespace)
    return namespace


def run_selected_block(namespace, args, parser):
    all_tasks = namespace.get("ALL_TASKS")
    if not all_tasks:
        parser.error("selected model code does not define ALL_TASKS")

    if args.task and args.task not in all_tasks:
        parser.error(
            f"unknown --task {args.task!r}; use one of: {', '.join(all_tasks)}"
        )

    tasks = [args.task] if args.task else all_tasks

    if args.status:
        namespace["show_status"]()
    elif args.block == 0:
        namespace["block0"]()
    elif args.block == 1:
        namespace["block1"](tasks)
    elif args.block == 2:
        namespace["block2"](tasks)
    elif args.block == 3:
        namespace["block3"](tasks)
    elif args.block == 4:
        namespace["block4"]()
    else:
        parser.print_help()


def run_all_models(args):
    for canonical in MODEL_SPECS:
        print("\n" + "=" * 72, flush=True)
        print(f"  Running {MODEL_SPECS[canonical]['label']} ({canonical})", flush=True)
        print("=" * 72, flush=True)

        cmd = [sys.executable, __file__, "--model", canonical]
        if args.status:
            cmd.append("--status")
        if args.block is not None:
            cmd.extend(["--block", str(args.block)])
        if args.task:
            cmd.extend(["--task", args.task])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode
    return 0


def main(argv=None):
    parser, args = parse_args(argv)

    if args.list_models:
        print_models()
        return 0

    if not args.model:
        if args.status or args.block is not None:
            return run_all_models(args)
        parser.error("--model is required unless --list-models is used")

    canonical = resolve_model(args.model, parser)
    namespace = load_model_namespace(canonical)
    run_selected_block(namespace, args, parser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
