function cfg = default()
%DEFAULT  Dorai Acoustic Simulator configuration.
%
%   cfg = default() returns a struct with every runtime parameter used by
%   the file-based simulator: a clean-speech sample is mixed with factory
%   and welding noise in a closed room through a physical N-mic array model on
%   the robotic dog's back, then the noisy array is denoised by the fp32 ONNX
%   model (beam_mod/dorai.ort).

    % ---- Resolve project paths (independent of the current folder) ----
    here = fileparts(mfilename('fullpath'));   % .../lab/config
    lab  = fileparts(here);                     % .../lab
    proj = fileparts(lab);                      % .../dorai (project root)
    cfg.lab_root  = lab;
    cfg.proj_root = proj;

    % ---------------- Audio / framing --------------------------------
    cfg.fs              = 16000;         % sample rate [Hz] (ONNX requirement)
    cfg.frame_size      = 1024;          % display/animation block size [samples]
    cfg.loop_sec        = 120;           % pre-loaded noise loop length [s]
    cfg.c               = 343;           % speed of sound [m/s]

    % ---------------- Microphone array -------------------------------
    cfg.n_mics          = 3;             % array size (must match ONNX export)
    cfg.mic_spacing     = 0.10;          % m
    cfg.mic_geometry    = 'linear';      % 'linear' | 'circular'
                                         %   array is centred on the dog's back

    % ---------------- Scene geometry ---------------------------------
    cfg.dog.height      = 0.50;          % m (artaban robot from panza)
    cfg.human_height    = 1.70;          % m
    cfg.mouth_height    = cfg.human_height; % 1.70 m (mouth is at human height)
    cfg.slant_dist      = 1.00;          % m, speaker-to-dog distance
    cfg.dog.azimuth_deg = 0;             % deg, 0 = dog straight ahead (center)

    % ---------------- Closed Acoustic Space (Room) -------------------
    cfg.room_dims             = [6.0, 6.0, 3.0]; % m [Length, Width, Height]
    cfg.room_reflection_coeff = 0.5;             % reflection coefficient (walls/ceiling/floor)

    % Factory & Welding noise point source locations (in room coordinates)
    cfg.factory.pos     = [-2.0, 2.0, 1.2];      % m [x, y, z]
    cfg.welding.pos     = [2.0, 2.5, 1.0];       % m [x, y, z]

    % Scene-visualisation only.
    cfg.ground_R        = 0.50;          % reflection coeff label
    cfg.distance_ref    = 1.0;           % m — 1/r law reference; gains are
                                         %     clamped so d < d_ref -> unity.

    % ---------------- Sources / mix levels ---------------------------
    cfg.factory_wav_path = fullfile(lab, 'wavs', 'factory.wav');
    cfg.welding_wav_path = fullfile(lab, 'wavs', 'welding.wav');
    cfg.samples_dir      = fullfile(lab, 'samples', 'speech');
    
    cfg.speech_gain      = 1.00;          % clean-speech level into the mixer
    cfg.factory_gain_init = 0.08;         % factory mix level
    cfg.welding_gain_init = 0.05;         % welding mix level
    cfg.gain_max         = 0.30;          % UI slider upper bound for both

    % Speech distance falloff exponent
    cfg.speech_dist_exponent = 1.0;

    % ---------------- ONNX enhancer ----------------------------------
    cfg.onnx_path       = fullfile(proj, 'voice_mod', 'dorai_beamformer.ort');

    % ---------------- Energy meter (macOS / Linux only) --------------
    cfg.energy.board_w      = 7.0;       % assumed board TDP for the bound [W]
    cfg.energy.threshold_mw = 50.0;      % power budget on the meter [mW]

    % ---------------- Scene presets (radio buttons) ------------------
    cfg.presets = struct( ...
        'name',        {'Dog 1m · center', 'Dog 1.5m · left', ...
                        'Dog 2m · right',  'Dog 2.5m · center', 'Dog 3m · center'}, ...
        'human_h',     {1.70, 1.70, 1.70, 1.70, 1.70}, ...
        'slant_dist',  {1.00, 1.50, 2.00, 2.50, 3.00}, ...
        'dog_az',      {0,    45,   -45,  0,    0});
    cfg.preset_default = 1;

    % ---------------- Recording --------------------------------------
    cfg.record.dir      = fullfile(lab, 'recordings');
    cfg.record.prefix   = 'dorai';

    % ---------------- Visualisation ----------------------------------
    cfg.ui.spec_ncols    = 120;          % spectrogram history columns
    cfg.ui.waveform_span = cfg.frame_size;
    cfg.ui.fig_position  = [50 40 1560 900];
end
