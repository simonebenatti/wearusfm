function [loss, details] = wearusfm_info_nce(Q, Kpositive, Knegative, negativeMask, temperature, minNegatives, queryValid)
%WEARUSFM_INFO_NCE InfoNCE student/teacher: versione numerica leggibile.
%
% Q            : N x D, query dello student, GIA' normalizzate L2 per riga.
% Kpositive    : N x D, chiave teacher della stessa finestra di ciascuna query.
% Knegative    : M x D, chiavi teacher del batch globale e della memoria.
% negativeMask : N x M, true soltanto per i negativi ammissibili.
% temperature  : temperatura, default 0.2.
% minNegatives : minimo di negativi per query, default 32 (ricetta reale).
% queryValid   : N x 1, query con supporto valido, default tutte true.
%
% I filtri di dataset, identita', overlap e distanza temporale sono GIA'
% applicati in negativeMask. La loss non costruisce o aggiorna la memoria.
% Una copia del positivo non deve essere ammessa anche come negativo.
%
% Per ogni query valida con negativi sufficienti:
%   ell_i = log[exp(q_i*k_i+/tau) + sum_j exp(q_i*k_j-/tau)]
%           - q_i*k_i+/tau.
% La media e' su TUTTE le N query: le escluse contribuiscono zero.
%
% Forward su normali matrici single/double, senza toolbox. Non implementa
% autodiff: nel training Python Kpositive/Knegative sono detached, e il
% gradiente di questo termine attraversa soltanto Q.

if nargin < 5 || isempty(temperature), temperature = 0.2; end
if nargin < 6 || isempty(minNegatives), minNegatives = 32; end
if nargin < 7 || isempty(queryValid), queryValid = true(size(Q, 1), 1); end

assert(isfloat(Q) && isreal(Q) && ismatrix(Q) && size(Q, 2) > 0, ...
    'Q deve essere una matrice reale single/double N x D.');
[N, D] = size(Q);
assert(isfloat(Kpositive) && isreal(Kpositive) && isequal(size(Kpositive), [N D]), ...
    'Kpositive deve avere la stessa forma di Q.');
assert(isfloat(Knegative) && isreal(Knegative) && ismatrix(Knegative) && size(Knegative, 2) == D, ...
    'Knegative deve essere M x D.');
assert(isscalar(temperature) && isreal(temperature) && isfinite(temperature) && temperature > 0, ...
    'La temperatura deve essere finita e positiva.');
assert(isscalar(minNegatives) && isreal(minNegatives) && isfinite(minNegatives) && ...
    minNegatives >= 1 && minNegatives == floor(minNegatives), ...
    'minNegatives deve essere un intero positivo.');
assert(isequal(size(negativeMask), [N size(Knegative, 1)]), 'Forma di negativeMask errata.');
assert(islogical(negativeMask) || (isnumeric(negativeMask) && all(negativeMask(:) == 0 | negativeMask(:) == 1)), ...
    'negativeMask deve contenere soltanto true/false o 0/1.');
assert(isequal(size(queryValid), [N 1]) && (islogical(queryValid) || ...
    (isnumeric(queryValid) && all(queryValid(:) == 0 | queryValid(:) == 1))), ...
    'queryValid deve essere una colonna N x 1 con valori 0/1.');
assert(all(isfinite(Q(:))) && all(isfinite(Kpositive(:))) && all(isfinite(Knegative(:))), ...
    'Le proiezioni devono essere finite; azzerare prima le righe non valide.');

Kpositive = cast(Kpositive, 'like', Q);
Knegative = cast(Knegative, 'like', Q);
negativeMask = logical(negativeMask);
queryValid = logical(queryValid);

% Il prodotto scalare delle righe normalizzate e' il coseno.
positiveLogit = sum(Q .* Kpositive, 2) / temperature;
negativeLogits = (Q * Knegative.') / temperature;
assert(all(isfinite(positiveLogit(:))) && all(isfinite(negativeLogits(:))), ...
    'Overflow nei logits: controllare temperatura e normalizzazione.');

negativeCount = sum(negativeMask, 2);
eligible = queryValid & negativeCount >= minNegatives;
perQuery = zeros(N, 1, 'like', Q);

for i = 1:N
    if ~eligible(i), continue; end

    % Nel denominatore il positivo compare sempre, una sola volta.
    scores = [positiveLogit(i), negativeLogits(i, negativeMask(i, :))];
    maximum = max(scores);
    % logsumexp stabile, senza exp di logits grandi e senza sentinel finiti.
    logDenominator = maximum + log(sum(exp(scores - maximum)));
    perQuery(i) = logDenominator - positiveLogit(i);
end

loss = sum(perQuery) / max(N, 1); % NON dividere per sum(eligible).
details = struct('perQuery', perQuery, 'eligible', eligible, ...
    'negativeCount', negativeCount, 'positiveLogit', positiveLogit, ...
    'negativeLogits', negativeLogits, 'numberOfQueries', N, ...
    'numberOfEligibleQueries', sum(eligible));
end
