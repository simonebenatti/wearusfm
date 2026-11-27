# Controprova R3: ricevute di validazione e checkout fuori dalla raccolta

Esito: il rilievo regge nei fatti. La gravità corretta è "minore", non "importante".

## Verifiche fatte
- Protocollo (worktree di release, docs/contrastive_cn_restart_probe_20261121.md, paragrafo "3-launcher autentica..."): la frase "trattiene stdout, stderr e ricevute" c'è davvero. Più sopra il protocollo dice anche "Conservare l'output" della quota.
- Sequenza, passo 5 (righe 120-121): validate_only.out.txt ed .err.txt finiscono in ~/wearusfm_local/cn_restart545_20261127, fuori dal repository.
- Sequenza, passo 7 (righe 169-191): l'elenco comprende report o execution_failure, i tempi delle fasi, allocation.json, i file delle due gambe, i due log SLURM del job GPU, il sacct dei due job e un ls -lR. Mancano:
  - validate_only;
  - il log campaign_checkout_<jobid>.out;
  - le uscite della quota (al passo 3, riga 90, "Conservare l'uscita" non ha un percorso);
  - l'ls-remote del deploy;
  - gli state.json;
  - lo sha256 dei file copiati.
  Il passo 7 chiede lo sha256 solo per i due log SLURM.
- Il brief (riga 21) ripete lo stesso perimetro.
- Gli archivi precedenti della linea 3 contengono questi file. Per esempio:
  - reloads_execution_20261121 ha CN_A.state.json, CN_A_login_check.txt e CN_A_operational_snapshot.txt;
  - readout_execution_20261121 ha campaign_checkout_65035600.out, checkout.state.json, deploy_verification.txt e local_files_sha256.json.

## Perché la gravità è minore
- L'omissione non fa fallire il lancio e non brucia un tentativo, perché la raccolta avviene dopo l'esecuzione.
- Non porta l'operatore fuori strada sull'esito.
- Le uscite di validate_only sono comunque trattenute sul Mac. Gli state.json restano in ~/wearusfm_local/jobs/ e il log del checkout resta su WORK. Si possono quindi recuperare e archiviare dopo.
- Resta una lacuna di completezza rispetto al protocollo e alla prassi degli archivi precedenti, che conviene correggere con la voce proposta.

## Non verificato
- Se l'ls-remote del passo 2 e l'uscita della quota verrebbero di fatto conservate dall'operatore. Nessun testo lo dice.
