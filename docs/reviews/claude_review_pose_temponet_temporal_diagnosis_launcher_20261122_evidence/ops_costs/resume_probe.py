"""Prova CPU, codice congelato vero, latenti sintetici: riprendere uno stato salvato (come current_states) e proseguire
fino al tetto della ricetta da' la stessa traiettoria di un run continuo? E oltre il tetto che cosa succede?"""
import json, time
import numpy as np, torch
from pose_temponet import train as T, heads, metrics as M

torch.set_num_threads(1)
t0 = time.time()
rng = np.random.default_rng(7)
n = 12
x = torch.from_numpy(rng.normal(size=(n, 160, 384)).astype(np.float16))
y = torch.from_numpy(rng.normal(size=(n, 160, 20)))
support = torch.ones((n, 148, 20), dtype=torch.bool)
data = T.TensorTrainingSet(x, y, support, [f"w{i:02d}" for i in range(n)])
rec = T._TestRecipe(initial_epochs=3, maximum_epochs=6, last_epochs=1, batch_size=4)
factory = lambda seed, fold: heads.make_head("TCN_post", seed=seed)
out = {}
# run continuo fino a 6 epoche
a = T.Trajectory(factory, data, lr=0.003, seed=0, fold=0, device="cpu", _recipe=rec)
a.train_to_step(6 * a.steps_per_epoch)
# run fermato a 3 epoche, stato salvato (passa da torch.save/torch.load come selection_states.pt), ripreso in un oggetto nuovo
b = T.Trajectory(factory, data, lr=0.003, seed=0, fold=0, device="cpu", _recipe=rec)
b.train_to_step(3 * b.steps_per_epoch)
import io
buf = io.BytesIO(); torch.save(b.state(), buf); buf.seek(0)
saved = torch.load(buf, map_location="cpu", weights_only=False)
c = T.Trajectory(factory, data, lr=0.003, seed=0, fold=0, device="cpu", _recipe=rec)
c.restore(saved)
c.train_to_step(6 * c.steps_per_epoch)
out["steps_per_epoch"] = a.steps_per_epoch
out["resumed_equals_continuous_payload_sha256"] = a.state()["payload_sha256"] == c.state()["payload_sha256"]
out["resumed_equals_continuous_model_sha256"] = heads.state_sha256(a.model) == heads.state_sha256(c.model)
out["epoch_losses_equal"] = a.epoch_losses == c.epoch_losses
try:
    c.train_to_step(6 * c.steps_per_epoch + 1)
    out["beyond_maximum"] = "accepted"
except ValueError as e:
    out["beyond_maximum"] = "ValueError: " + str(e)
# ricetta di produzione: tetto 300 epoche; un orizzonte diverso viene rifiutato
try:
    T._check_recipe(T.Recipe(maximum_epochs=600)); out["production_recipe_600"] = "accepted"
except ValueError as e:
    out["production_recipe_600"] = "ValueError: " + str(e)
p = T.Trajectory(factory, data, lr=0.003, seed=0, fold=0, device="cpu")
try:
    p.train_to_step(300 * p.steps_per_epoch + 1); out["production_beyond_300"] = "accepted"
except ValueError as e:
    out["production_beyond_300"] = "ValueError: " + str(e)
out["elapsed_seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
