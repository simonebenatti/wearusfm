function [loss, details] = wearusfm_rank_contrast(H, Y, windowId, valid, pairAllowed, temperature)
%WEARUSFM_RANK_CONTRAST Forward numerico della RnC corretta di WearUsFM.
%   H: N x D, readout mean_K per istante, gia' selezionati; nessun pooling
%      temporale. In Python stride=8, phase=4 seleziona i patch 5,13,21,...
%      (indici MATLAB), poi tiene solo quelli fisici con almeno un joint valido.
%   Y: N x J, angoli gia' standardizzati; NaN nei joint non validi sono ignorati.
%   windowId: N x 1, identificatori interi positivi delle finestre.
%   valid: N x J, maschera logica (o numerica 0/1) dei joint validi.
%   pairAllowed: N x N simmetrica; [] = tutte le coppie tranne la diagonale.
%   temperature: scalare positivo, default 0.1 (anche se passato come []).
%
%   Tutti i candidati ammessi sono anche positivi. Per ogni positivo j,
%   il denominatore comprende i candidati k con distanza(i,k)>=distanza(i,j),
%   incluse le parita'. La stessa maschera vale in entrambi i ruoli.
%   Ogni anchor e' divisa per la propria loss a similarita' uniformi;
%   contano solo anchor con almeno due candidati. Si media prima per finestra,
%   poi sulle finestre con almeno un anchor valido. Il caso vuoto restituisce 0.
%
%   details: distances, logits, candidateMask (N x N); anchorLoss,
%   anchorValid, reference, rawSum (N x 1); windowIds e windowLoss.
%   windowIds contiene gli ID presenti, ordinati; anchorLoss e windowLoss
%   sono 0 dove manca supporto. rawSum/reference sono somme per anchor,
%   prima della normalizzazione; reference=sum_j log(|denominatore_j|).
%   distances e' double; le distanze senza joint condivisi sono 0 e mascherate.
%
%   Solo MATLAB base, array single/double, senza autograd o dlarray.
%   H mantiene la sua precisione; i target sono confrontati in double.
%   Servono tutti i candidati del batch globale: medie di batch separati
%   non riproducono questa loss. Qui resolution=0, senza quantizzazione.

narginchk(4, 6);
if nargin < 5
    pairAllowed = [];
end
if nargin < 6 || isempty(temperature)
    temperature = 0.1;
end
if ~(isa(H, 'double') || isa(H, 'single')) || ~isreal(H) || ...
        ~ismatrix(H) || issparse(H) || size(H, 2) < 1 || any(~isfinite(H(:)))
    error('wearusfm_rank_contrast:H', 'H deve essere una matrice single/double finita N x D, D>=1.');
end
N = size(H, 1);
if ~(isa(Y, 'double') || isa(Y, 'single')) || ~isreal(Y) || ...
        ~ismatrix(Y) || issparse(Y) || size(Y, 1) ~= N || size(Y, 2) < 1
    error('wearusfm_rank_contrast:Y', 'Y deve essere una matrice single/double N x J, J>=1.');
end
if ~isnumeric(windowId) || ~isreal(windowId) || ...
        ~isequal(size(windowId), [N, 1]) || any(~isfinite(windowId(:))) || ...
        any(windowId(:) < 1) || any(windowId(:) ~= fix(windowId(:)))
    error('wearusfm_rank_contrast:windowId', 'windowId deve contenere N x 1 interi positivi.');
end
valid = binary_mask(valid, size(Y), 'valid');
if any(~isfinite(Y(valid)))
    error('wearusfm_rank_contrast:Y', 'I target dei joint validi devono essere finiti.');
end
if isempty(pairAllowed)
    pairAllowed = true(N, N);
else
    pairAllowed = binary_mask(pairAllowed, [N, N], 'pairAllowed');
    if ~isequal(pairAllowed, pairAllowed.')
        error('wearusfm_rank_contrast:pairAllowed', 'pairAllowed deve essere simmetrica.');
    end
end
if ~isnumeric(temperature) || ~isreal(temperature) || ...
        ~isscalar(temperature) || ~isfinite(temperature) || temperature <= 0
    error('wearusfm_rank_contrast:temperature', 'temperature deve essere finita e positiva.');
end

% LayerNorm senza parametri: varianza di popolazione, poi norma L2.
centered = H - mean(H, 2);
U = centered ./ sqrt(mean(centered .^ 2, 2) + cast(1e-5, 'like', H));
U = U ./ max(sqrt(sum(U .^ 2, 2)), cast(1e-12, 'like', H));
logits = (U * U.') ./ cast(temperature, 'like', H);

% RMS solo sui joint validi in entrambi gli istanti; non si legge Y altrove.
distances = zeros(N, N);
candidateMask = false(N, N);
for i = 1:N
    for j = (i + 1):N
        shared = valid(i, :) & valid(j, :);
        if any(shared)
            delta = double(Y(i, shared)) - double(Y(j, shared));
            distances(i, j) = sqrt(mean(delta .^ 2));
            distances(j, i) = distances(i, j);
            candidateMask(i, j) = pairAllowed(i, j);
            candidateMask(j, i) = pairAllowed(j, i);
        end
    end
end
if any(~isfinite(logits(:))) || any(~isfinite(distances(candidateMask)))
    error('wearusfm_rank_contrast:overflow', 'Logit o distanza candidata non finiti: controllare scala e temperatura.');
end

rawSum = zeros(N, 1, 'like', H);
reference = zeros(N, 1, 'like', H);
anchorLoss = zeros(N, 1, 'like', H);
anchorValid = sum(candidateMask, 2) >= 2;
for i = 1:N
    candidates = find(candidateMask(i, :));
    for j = candidates
        denominator = candidates(distances(i, candidates) >= distances(i, j));
        scores = logits(i, denominator);
        peak = max(scores);
        % logsumexp sui soli candidati reali; nessun valore sentinella.
        % La sottrazione anticipata del logit evita cancellazioni inutili.
        term = (peak - logits(i, j)) + log(sum(exp(scores - peak)));
        rawSum(i) = rawSum(i) + term;
        reference(i) = reference(i) + log(cast(numel(denominator), 'like', H));
    end
    if anchorValid(i)
        anchorLoss(i) = rawSum(i) / reference(i);
    end
end

% Ogni finestra valida ha lo stesso peso, anche con numeri diversi di anchor.
windowIds = unique(windowId);
windowLoss = zeros(numel(windowIds), 1, 'like', H);
windowValid = false(numel(windowIds), 1);
for w = 1:numel(windowIds)
    anchors = (windowId == windowIds(w)) & anchorValid;
    windowValid(w) = any(anchors);
    if windowValid(w)
        windowLoss(w) = mean(anchorLoss(anchors));
    end
end
loss = zeros(1, 'like', H);
if any(windowValid)
    loss = mean(windowLoss(windowValid));
end

details = struct('distances', distances, 'logits', logits, ...
    'candidateMask', candidateMask, 'anchorLoss', anchorLoss, ...
    'anchorValid', anchorValid, 'reference', reference, 'rawSum', rawSum, ...
    'windowIds', windowIds, 'windowLoss', windowLoss);
end

function mask = binary_mask(mask, expectedSize, name)
% Accetta solo maschere logiche o numeriche esplicitamente binarie.
if ~(islogical(mask) || isnumeric(mask)) || ~isreal(mask) || ...
        ~isequal(size(mask), expectedSize) || any(mask(:) ~= 0 & mask(:) ~= 1)
    error('wearusfm_rank_contrast:mask', '%s deve essere una maschera binaria della dimensione prevista.', name);
end
mask = logical(mask);
end
