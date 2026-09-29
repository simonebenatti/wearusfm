function convert_camargo_mcos_tables(in_root, out_root)
% Converte i file .mat di Camargo 2021 (tabelle MATLAB MCOS, illeggibili da
% scipy.io.loadmat/h5py/pymatreader/mat73 - verificato in sessione, 29/09/2026,
% nessun tool Python le supporta) in file .mat "-v7" semplici: due variabili,
% `colnames` (cell array di stringhe) e `data` (matrice numerica double),
% nessun oggetto MCOS - leggibile da scipy.io.loadmat senza problemi.
%
% Uso (non interattivo):
%   matlab -batch "convert_camargo_mcos_tables('/path/in', '/path/out')"
%
% Mantiene la struttura di directory relativa fra in_root e out_root.
% Un file che fallisce la conversione viene loggato e saltato, non ferma il
% batch (migliaia di file, un singolo errore non deve fermare tutto).

files = dir(fullfile(in_root, '**', '*.mat'));
fprintf('trovati %d file .mat sotto %s\n', numel(files), in_root);

n_ok = 0;
n_failed = 0;
failures = {};

for i = 1:numel(files)
    in_path = fullfile(files(i).folder, files(i).name);
    rel_path = erase(in_path, [in_root filesep]);
    out_path = fullfile(out_root, rel_path);

    try
        s = load(in_path);
        fn = fieldnames(s);
        if numel(fn) ~= 1
            error('atteso 1 variabile nel file, trovate %d: %s', numel(fn), strjoin(fn, ', '));
        end
        T = s.(fn{1});
        if ~istable(T)
            error('la variabile %s non e'' una tabella (class=%s)', fn{1}, class(T));
        end

        colnames = T.Properties.VariableNames;
        data = table2array(T);
        if ~isnumeric(data)
            error('table2array non ha prodotto una matrice numerica (class=%s)', class(data));
        end

        out_dir = fileparts(out_path);
        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end
        save(out_path, 'colnames', 'data', '-v7');
        n_ok = n_ok + 1;
    catch err
        n_failed = n_failed + 1;
        failures{end+1} = sprintf('%s: %s', rel_path, err.message); %#ok<AGROW>
        fprintf('FALLITO %s: %s\n', rel_path, err.message);
    end

    if mod(i, 200) == 0
        fprintf('[%d/%d] ok=%d fail=%d\n', i, numel(files), n_ok, n_failed);
    end
end

fprintf('=== fine: ok=%d fail=%d su %d file ===\n', n_ok, n_failed, numel(files));
if n_failed > 0
    fprintf('--- elenco fallimenti ---\n');
    for i = 1:numel(failures)
        fprintf('%s\n', failures{i});
    end
end
end
