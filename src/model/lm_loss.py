"""Memory-efficient causal-LM loss.

`AutoModelForCausalLM(labels=...)` materialises logits of shape [B, L, vocab]
for the whole sequence even though cross-entropy only reads the positions whose
label is not IGNORE_INDEX (the short answer span). For 8k-token prompts and a
260k vocab that single tensor chain costs tens of GB.

`causal_lm_loss_on_labels` runs the transformer body on the full sequence
(so the answer tokens still attend to the entire prompt and gradients still
flow back through it), then applies the LM head + cross-entropy only at the
supervised positions. This is numerically equivalent to the HF loss:

  * same shift: the hidden state at position i predicts the label at i+1
    (HF computes logits[:, :-1] vs labels[:, 1:]);
  * same reduction: mean over non-ignored label tokens;
  * same fp32 upcast of logits before cross-entropy;
  * positions with IGNORE_INDEX contribute zero loss and zero gradient in
    both formulations, so skipping their logits changes nothing.
"""

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _get_decoder(model):
    if hasattr(model, "get_decoder"):
        decoder = model.get_decoder()
        if decoder is not None:
            return decoder
    return model.model


def _logit_softcap(model):
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap is None and hasattr(model.config, "text_config"):
        cap = getattr(model.config.text_config, "final_logit_softcapping", None)
    return cap


def causal_lm_loss_on_labels(model, *, attention_mask, labels,
                             input_ids=None, inputs_embeds=None):
    """Drop-in replacement for `model(..., labels=labels).loss`."""
    hidden = _get_decoder(model)(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        return_dict=True,
    ).last_hidden_state

    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    keep = shift_labels != IGNORE_INDEX

    logits = model.get_output_embeddings()(shift_hidden[keep])
    cap = _logit_softcap(model)
    if cap:
        logits = torch.tanh(logits / cap) * cap

    return F.cross_entropy(logits.float(), shift_labels[keep], reduction="mean")
