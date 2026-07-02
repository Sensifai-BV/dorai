function run_simulation()
%RUN_SIMULATION  Dorai Acoustic Simulator entry point.
%
%   Adds the project folders to the MATLAB path, loads the default config,
%   points MATLAB's Python at the lab/.pyenv venv (created by run_dorai.sh),
%   builds the scene geometry, wires up the audio manager + ONNX enhancer,
%   and launches the UI.
%
%   Edit config/default.m to reshape any runtime parameter.

    setup_paths();

    cfg = default();
    setup_python(cfg);
    geo = build_geometry(cfg);
    print_geometry(geo);

    audio    = AudioIO(cfg);
    enhancer = OrtEnhancer(cfg.onnx_path, cfg.lab_root);

    ui = SimulatorUI(cfg, geo, audio, enhancer);
    ui.start();
end


function setup_paths()
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'config'));
    addpath(fullfile(here, 'core'));
    addpath(fullfile(here, 'visualization'));
end


function setup_python(cfg)
%SETUP_PYTHON  Point MATLAB's pyenv at lab/.pyenv if that venv exists.
%   Must run before any py.* call. pyenv can only be switched while the
%   interpreter is NotLoaded (i.e. early in a fresh MATLAB session, which
%   is exactly how run_dorai.sh launches this script).
    venv = fullfile(cfg.lab_root, '.pyenv', 'bin', 'python3');
    if ~isfile(venv)
        venv = fullfile(cfg.lab_root, '.pyenv', 'bin', 'python');
    end
    if ~isfile(venv)
        fprintf(['[Dorai] No lab/.pyenv venv found — using MATLAB''s ' ...
                 'default Python. Run lab/run_dorai.sh to create it.\n']);
        return;
    end
    try
        pe = pyenv;
        if ~strcmp(char(pe.Executable), venv)
            if strcmp(char(pe.Status), 'NotLoaded')
                pyenv('Version', venv, 'ExecutionMode', 'OutOfProcess');
                fprintf('[Dorai] Python -> %s (OutOfProcess)\n', venv);
            else
                warning(['[Dorai] Python already loaded (%s); restart MATLAB ' ...
                         'to switch to lab/.pyenv.'], char(pe.Executable));
            end
        else
            fprintf('[Dorai] Python already set to venv: %s\n', venv);
        end
    catch ME
        warning('[Dorai] Could not configure pyenv: %s', ME.message);
    end
end
