import numpy as np

from wearusfm.data.consumer import compute_asse_m_metrics, required_windows_per_s


def test_waiting_fraction_zero_when_always_faster_than_t_passo():
    fetch = np.full(100, 0.005)  # 5 ms, sempre sotto t_passo
    m = compute_asse_m_metrics(fetch, t_passo_s=0.05)
    assert m.waiting_fraction == 0.0


def test_waiting_fraction_one_when_dataloader_never_delivers_in_time():
    # tempo di parete sempre = fetch (mai t_passo, perche' fetch >> t_passo)
    fetch = np.full(100, 1.0)
    m = compute_asse_m_metrics(fetch, t_passo_s=0.01)
    assert m.waiting_fraction > 0.95


def test_multi_rank_takes_max_across_ranks():
    # il risultato multi-rank deve coincidere esattamente con quello di un caso
    # 1D dove il tempo per batch e' gia' il massimo sui rank: prova diretta della
    # riduzione "massimo sui rank" della barriera (spec §7), non solo una percentile
    rng = np.random.default_rng(0)
    fetch_4rank = rng.uniform(0.005, 0.5, size=(4, 50))
    fetch_prereduced = fetch_4rank.max(axis=0)

    m_multi = compute_asse_m_metrics(fetch_4rank, t_passo_s=0.05)
    m_single = compute_asse_m_metrics(fetch_prereduced, t_passo_s=0.05)

    assert m_multi == m_single


def test_required_windows_per_s_matches_spec_order_of_magnitude():
    # spec §9: ~70 finestre/s a 30M, ~20 a 100M (K=64, contesto 4s, patch 25ms, MFU 40%)
    r30m = required_windows_per_s(30e6)
    r100m = required_windows_per_s(100e6)
    assert 60 <= r30m <= 80
    assert 15 <= r100m <= 25
    assert r30m > r100m  # il modello piu' piccolo e' il caso vincolante (piu' richiesta)


def test_t_passo_scales_with_batch_size_not_per_window():
    # bug reale trovato sul primo run Leonardo: confrontare il tempo di un BATCH contro
    # la soglia di UNA finestra sovrastima l'attesa. Un batch che consegna 64 finestre
    # in un tempo pari a 64 * t_passo (cioe' esattamente al ritmo richiesto) non deve
    # segnalare attesa, ne' con batch_size=1 ne' con batch_size=64.
    t_passo_s = 0.01  # tempo richiesto per UNA finestra
    batch_size = 64
    fetch = np.full(50, t_passo_s * batch_size)  # il batch arriva esattamente in tempo
    m = compute_asse_m_metrics(fetch, t_passo_s, batch_size=batch_size)
    assert m.waiting_fraction == 0.0
    assert abs(m.windows_per_s - batch_size / (t_passo_s * batch_size)) < 1e-9


def test_windows_per_s_accounts_for_batch_size():
    fetch = np.full(50, 0.25)  # 0.25s per batch
    m1 = compute_asse_m_metrics(fetch, t_passo_s=0.01, batch_size=1)
    m64 = compute_asse_m_metrics(fetch, t_passo_s=0.01, batch_size=64)
    assert abs(m64.windows_per_s - 64 * m1.windows_per_s) < 1e-6
