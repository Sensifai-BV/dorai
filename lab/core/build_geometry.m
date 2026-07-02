function geo = build_geometry(cfg)
%BUILD_GEOMETRY  Compute scene geometry, image sources, per-path TDOAs and gains.
%
%   geo = build_geometry(cfg)
%
%   Returns a geometry struct for Dorai Acoustic Simulator.
%   Uses the Image Source Method (1st order reflections: floor, ceiling, 4 walls)
%   to model a closed room (closed acoustic space) for speech, factory, and
%   welding noise sources.

    c     = cfg.c;
    mh    = cfg.mouth_height;
    d_ref = cfg.distance_ref;
    fs    = cfg.fs;

    % --- Human mouth position (origin in xy, mouth height in z) -------
    pos_human   = [0.0, 0.0, mh];

    % --- Dog position (receiver center) --------------------------------
    dog_az = local_dog_azimuth_deg_(cfg);
    pos_dog = [cfg.slant_dist * cosd(dog_az), cfg.slant_dist * sind(dog_az), cfg.dog.height];

    % --- Mic array centred on the dog ---------------------------------
    pos_mics = build_mic_array_(pos_dog, cfg.n_mics, ...
                                cfg.mic_spacing, mic_geometry_(cfg));

    % --- Factory-noise and Welding-noise positions ---------------------
    pos_factory = cfg.factory.pos;
    pos_welding = cfg.welding.pos;

    % --- Closed room reflection coefficients & image sources ----------
    R_coeff = cfg.room_reflection_coeff;
    R_weights = [1.0, repmat(R_coeff, 1, 6)]; % path 1: direct, paths 2..7: reflections
    
    % Generate 7 image source positions for each source
    imgs_speech  = get_image_sources_(pos_human,   cfg.room_dims);
    imgs_factory = get_image_sources_(pos_factory, cfg.room_dims);
    imgs_welding = get_image_sources_(pos_welding, cfg.room_dims);

    % --- Per-path per-mic distances -----------------------------------
    % Arrays of size [7 x n_mics]
    d_speech  = zeros(7, cfg.n_mics);
    d_factory = zeros(7, cfg.n_mics);
    d_welding = zeros(7, cfg.n_mics);
    
    for p = 1:7
        d_speech(p, :)  = vecnorm(pos_mics - imgs_speech(p, :), 2, 2).';
        d_factory(p, :) = vecnorm(pos_mics - imgs_factory(p, :), 2, 2).';
        d_welding(p, :) = vecnorm(pos_mics - imgs_welding(p, :), 2, 2).';
    end

    % --- Per-path TDOAs (delays relative to direct path first-mic arrival) ---
    [delays_speech,  frac_delays_speech]  = path_delays_(d_speech,  c, fs);
    [delays_factory, frac_delays_factory] = path_delays_(d_factory, c, fs);
    [delays_welding, frac_delays_welding] = path_delays_(d_welding, c, fs);

    % --- Per-path gains (including 1/r spreading and reflection loss) --
    gains_speech  = zeros(7, cfg.n_mics);
    gains_factory = zeros(7, cfg.n_mics);
    gains_welding = zeros(7, cfg.n_mics);
    
    alpha = speech_dist_exponent_(cfg);
    for p = 1:7
        gains_speech(p, :)  = R_weights(p) * (d_ref ./ max(d_speech(p, :), d_ref)) .^ alpha;
        gains_factory(p, :) = R_weights(p) * (d_ref ./ max(d_factory(p, :), d_ref));
        gains_welding(p, :) = R_weights(p) * (d_ref ./ max(d_welding(p, :), d_ref));
    end

    % --- Legacy fields for drawing or diagnostic functions -----------
    geo.pos_human         = pos_human;
    geo.pos_dog           = pos_dog;
    geo.pos_drone         = pos_dog;          % legacy alias for UI/drawing
    geo.pos_mics          = pos_mics;
    geo.pos_factory       = pos_factory;
    geo.pos_welding       = pos_welding;
    geo.pos_env           = pos_welding;      % legacy alias for UI/drawing
    geo.drone_agl         = pos_dog(3);       % legacy alias for UI
    
    geo.dist_speech       = d_speech(1, :).';  % direct path distance for mic rows
    geo.dist_factory      = d_factory(1, :).';
    geo.dist_welding      = d_welding(1, :).';
    
    % Delays for speech direct path (legacy diagnostic print)
    geo.delays            = delays_speech(1, :).';
    geo.delays_speech     = delays_speech;
    geo.delays_drone      = delays_factory;   % legacy mapping
    geo.delays_env        = delays_welding;   % legacy mapping
    
    geo.frac_delays_speech  = frac_delays_speech;
    geo.frac_delays_factory = frac_delays_factory;
    geo.frac_delays_welding = frac_delays_welding;
    geo.frac_delays_drone   = frac_delays_factory; % legacy mapping
    geo.frac_delays_env     = frac_delays_welding; % legacy mapping

    geo.gains_speech      = gains_speech;
    geo.gains_factory     = gains_factory;
    geo.gains_welding     = gains_welding;
    geo.gains_drone       = gains_factory;    % legacy mapping
    geo.gains_env         = gains_welding;    % legacy mapping
    
    geo.distance_ref      = d_ref;
    geo.room_dims         = cfg.room_dims;
end

% --------------------------------------------------------------------
function a = speech_dist_exponent_(cfg)
%SPEECH_DIST_EXPONENT_  Distance-falloff exponent for the speech gain.
    a = 1.0;
    if isfield(cfg, 'speech_dist_exponent') && ~isempty(cfg.speech_dist_exponent)
        a = cfg.speech_dist_exponent;
    end
    a = max(0, min(1, a));
end

% --------------------------------------------------------------------
function [tau_int, tau_frac] = path_delays_(d, c, fs)
%PATH_DELAYS_  Compute sample delays for [7 x n_mics] distance matrix.
%   Delays are computed relative to the direct path (row 1) first arrival.
    tau_abs_frac = (d / c) * fs;
    min_direct   = min(tau_abs_frac(1, :));
    tau_frac     = tau_abs_frac - min_direct;
    tau_int      = round(tau_frac);
end

% --------------------------------------------------------------------
function pos_imgs = get_image_sources_(pos, room_dims)
%GET_IMAGE_SOURCES_  Compute 1 direct path + 6 first-order room reflection image sources.
%   room_dims: [Length, Width, Height]
%   Human is at [0, 0] in xy. Room bounds are:
%     x in [-L/2, L/2]
%     y in [-W/2, W/2]
%     z in [0, H]
    L = room_dims(1);
    W = room_dims(2);
    H = room_dims(3);
    
    x = pos(1);
    y = pos(2);
    z = pos(3);
    
    pos_imgs = zeros(7, 3);
    pos_imgs(1, :) = pos;                  % 1: Direct path
    pos_imgs(2, :) = [x, y, -z];           % 2: Floor (z = 0)
    pos_imgs(3, :) = [x, y, 2*H - z];      % 3: Ceiling (z = H)
    pos_imgs(4, :) = [-L - x, y, z];       % 4: Left wall (x = -L/2)
    pos_imgs(5, :) = [L - x, y, z];        % 5: Right wall (x = L/2)
    pos_imgs(6, :) = [x, -W - y, z];       % 6: Back wall (y = -W/2)
    pos_imgs(7, :) = [x, W - y, z];        % 7: Front wall (y = W/2)
end

% --------------------------------------------------------------------
function pos = build_mic_array_(center, n, spacing, geometry)
%BUILD_MIC_ARRAY_  Build linear or circular array centred on `center`.
    center = reshape(center, 1, 3);
    switch lower(geometry)
        case 'linear'
            % ULA along the x-axis, centred on the dog.
            idx     = (0:n-1).' - (n-1)/2;
            offsets = [idx * spacing, zeros(n,1), zeros(n,1)];
            pos     = center + offsets;
        case 'circular'
            % Equal-chord circular array; adjacent mic distance ≈ spacing.
            theta   = (0:n-1).' * 2*pi/n;
            radius  = spacing / (2*sin(pi/max(n,2)));
            offsets = [radius*cos(theta), radius*sin(theta), zeros(n,1)];
            pos     = center + offsets;
        otherwise
            error('build_geometry:UnknownGeometry', ...
                  'Unsupported mic geometry "%s" (use ''linear'' or ''circular'').', geometry);
    end
end

% --------------------------------------------------------------------
function g = mic_geometry_(cfg)
    if isfield(cfg, 'mic_geometry') && ~isempty(cfg.mic_geometry)
        g = cfg.mic_geometry;
    else
        g = 'linear';
    end
end

% --------------------------------------------------------------------
function az = local_dog_azimuth_deg_(cfg)
    az = 0;
    if isfield(cfg, 'dog') && isstruct(cfg.dog) ...
       && isfield(cfg.dog, 'azimuth_deg')
        az = cfg.dog.azimuth_deg;
    elseif isfield(cfg, 'presets') && isfield(cfg, 'preset_default')
        p = cfg.presets(cfg.preset_default);
        if isfield(p, 'dog_az')
            az = p.dog_az;
        end
    end
end
