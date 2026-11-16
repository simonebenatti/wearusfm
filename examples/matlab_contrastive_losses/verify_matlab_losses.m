function report = verify_matlab_losses()
%VERIFY_MATLAB_LOSSES Confronta il forward MATLAB con fixture del Python reale.
here = fileparts(mfilename('fullpath'));
reference = load(fullfile(here, 'python_reference.mat'));
tolerance = 5e-10;
maximumError = 0;
for c = 1:numel(reference.nceCases)
    f = reference.nceCases{c};
    [loss, details] = wearusfm_info_nce(f.Q, f.Kpositive, f.Knegative, ...
        logical(f.negativeMask), f.temperature, f.minNegatives, logical(f.queryValid));
    err = max(abs([double(loss) - f.expectedLoss; double(details.perQuery(:)) - f.expectedPerQuery(:)]));
    assert(err < tolerance, 'InfoNCE: differenza dal Python nel caso %d.', c);
    assert(isequal(details.eligible(:), logical(f.expectedEligible(:))));
    maximumError = max(maximumError, err);
end
for c = 1:numel(reference.rncCases)
    f = reference.rncCases{c};
    [loss, details] = wearusfm_rank_contrast(f.H, f.Y, f.windowId, ...
        logical(f.valid), logical(f.pairAllowed), f.temperature);
    err = abs(double(loss) - f.expectedLoss);
    assert(err < tolerance, 'RnC: differenza dal Python nel caso %d.', c);
    assert(sum(details.anchorValid) == f.expectedAnchors);
    assert(numel(unique(f.windowId(details.anchorValid))) == f.expectedWindows);
    assert(sum(sum(details.candidateMask(details.anchorValid, :))) == f.expectedPairs);
    maximumError = max(maximumError, err);
end
demo = demo_contrastive_losses(false);
report = struct('matlabVersion', version, 'nceCasesPassed', numel(reference.nceCases), ...
    'rncCasesPassed', numel(reference.rncCases), 'maximumAbsoluteError', maximumError, ...
    'tolerance', tolerance, 'demoPassed', true, 'demo', demo);
fprintf('Confronto Python/MATLAB: %d casi InfoNCE + %d casi RnC, errore massimo %.3g.\n', ...
    report.nceCasesPassed, report.rncCasesPassed, maximumError);
end
