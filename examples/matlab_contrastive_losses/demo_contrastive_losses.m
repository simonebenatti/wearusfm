function results = demo_contrastive_losses(makePlots)
%DEMO_CONTRASTIVE_LOSSES Esempi piccoli, da eseguire e ispezionare nel debugger.
%   results = demo_contrastive_losses;       % con figure
%   results = demo_contrastive_losses(false); % soltanto numeri
% I dati sono sintetici: illustrano le formule, non un risultato WearUsFM.
if nargin < 1, makePlots = true; end

%% 1. InfoNCE: query vicina al proprio positivo, lontana dagli altri.
Q = eye(3);                     % tre query gia' normalizzate L2
Kpositive = eye(3);             % teacher: stessa finestra sulla stessa riga
Knegative = eye(3);             % piccolo banco comune di chiavi teacher
negativeMask = ~eye(3);         % il positivo non ricompare come negativo
temperatureNCE = 0.2;
% SOLO in questo esempio usiamo 2 negativi: nella ricetta reale il minimo e' 32.
minimumToyNegatives = 2;
[nceAligned, nceDetails] = wearusfm_info_nce(Q, Kpositive, Knegative, ...
    negativeMask, temperatureNCE, minimumToyNegatives);
[nceMisaligned, ~] = wearusfm_info_nce(circshift(Q, 1, 2), Kpositive, Knegative, ...
    negativeMask, temperatureNCE, minimumToyNegatives);
uniform = ones(3) / sqrt(3);
nceUniform = wearusfm_info_nce(uniform, uniform, uniform, ...
    negativeMask, temperatureNCE, minimumToyNegatives);
[nceNoSupport, unsupported] = wearusfm_info_nce(Q, Kpositive, Knegative, negativeMask);
assert(abs(nceUniform - log(3)) < 1e-12);
assert(nceAligned < nceMisaligned);
assert(nceNoSupport == 0 && ~any(unsupported.eligible));

%% 2. RnC: angoli simili devono avere rappresentazioni piu' simili.
theta = (0:0.2:1).';
Y = [theta, 0.5 * theta];       % due angoli gia' standardizzati
H = [cos(theta), sin(theta), -cos(theta), -sin(theta)];
windowId = [1; 1; 2; 2; 2; 3]; % tre finestre, con numeri diversi di istanti
valid = true(size(Y));
pairAllowed = ~eye(numel(theta));
[rncOrdered, rncDetails] = wearusfm_rank_contrast(H, Y, windowId, valid, pairAllowed);
permutation = [1, 6, 2, 5, 3, 4];
rncShuffled = wearusfm_rank_contrast(H(permutation, :), Y, windowId, valid, pairAllowed);
rncUniform = wearusfm_rank_contrast(repmat(H(1, :), numel(theta), 1), ...
    Y, windowId, valid, pairAllowed);
assert(abs(rncUniform - 1) < 1e-12); % riferimento con similarita' tutte uguali
assert(rncOrdered < rncShuffled);

fprintf('\nInfoNCE: allineata %.6f; disallineata %.6f; uniforme %.6f (= log(3)).\n', ...
    nceAligned, nceMisaligned, nceUniform);
fprintf('RnC: ordinata %.6f; mescolata %.6f; uniforme %.6f (= 1).\n', ...
    rncOrdered, rncShuffled, rncUniform);
fprintf('InfoNCE con minimo reale 32: %.1f, ma ZERO query ammissibili: manca supporto.\n', nceNoSupport);

results = struct('nceAligned', nceAligned, 'nceMisaligned', nceMisaligned, ...
    'nceUniform', nceUniform, 'nceDetails', nceDetails, ...
    'rncOrdered', rncOrdered, 'rncShuffled', rncShuffled, ...
    'rncUniform', rncUniform, 'rncDetails', rncDetails);

if makePlots
    figure('Name', 'Due contrastive: esempi sintetici', 'Color', 'w');
    subplot(2, 2, 1);
    bar([nceAligned, nceMisaligned, nceUniform]);
    set(gca, 'XTickLabel', {'Allineata', 'Disallineata', 'Uniforme'});
    ylabel('InfoNCE'); title('Stessa finestra: positivo corretto');
    subplot(2, 2, 2);
    bar([rncOrdered, rncShuffled, rncUniform]);
    set(gca, 'XTickLabel', {'Ordinata', 'Mescolata', 'Uniforme'});
    ylabel('RnC normalizzata'); title('Ordine imposto dagli angoli');
    subplot(2, 2, 3);
    imagesc(rncDetails.distances); axis image; colorbar;
    title('Distanze RMS tra pose'); xlabel('Istante j'); ylabel('Istante i');
    subplot(2, 2, 4);
    imagesc(rncDetails.logits * 0.1); axis image; colorbar;
    title('Similarita'' tra rappresentazioni'); xlabel('Istante j'); ylabel('Istante i');
end
end
