# NeuroRVQ (codice di terzi, copiato per la replica del passo 5)

- **Origine:** https://github.com/KonstantinosBarmpas/NeuroRVQ, commit `926e770d9d16b6aa308404280fa0cc0211a6f9fb` (06/08/2026, «Update
  NeuroRVQ.py»), lo stesso clone usato per V1-V4 su Leonardo (`$SCRATCH/external/NeuroRVQ_926e770`).
- **Autori:** Barmpas et al., «NeuroRVQ: Multi-Scale Biosignal Tokenization for Generative Foundation Models», arXiv 2510.13068.
- **Licenza:** quella del repo originale, in `LICENSE` (CC BY-NC 4.0), con l'attribuzione agli autori. Entra nel repo per decisione di Simone del
  04/10/2026 («teniamo tutto dato che come abbiamo gia deciso non ci interessano le licenze»; decisioni d'uso sulle licenze del 30/09).
- **Cosa c'e':** solo la parte EMG (pacchetto del modello e del tokenizer, moduli e esempi d'inferenza, esempio di fine-tuning, esempio di
  preprocessing, flag). Niente pesi: i checkpoint restano su Leonardo in `$WORK/models/`.
- **Non modificato.** Le differenze fra questo codice e il protocollo del paper sono nel fatto 24 (`docs/fatti_da_verificare.md`); la replica le
  gestisce nei nostri script, non toccando questi file.
