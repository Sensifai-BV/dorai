classdef SourceMixer < handle
%SOURCEMIXER  Physical multi-source -> N-microphone array simulator.
%
%   Every microphone receives every source. Per-source per-mic
%   propagation applies fractional sample delays (linear interpolation)
%   and room reflections for a closed acoustic space:
%
%       mic_m[n] = sum over s in {speech, factory, welding}
%                      sum over path p in 1..7
%                          gain_s(p, m) * src_s[n - frac_delay_s(p, m)]
%
%   Fractional delays that exceed one block are handled with a per-source
%   history ring buffer, so block boundaries do not click.

    properties
        cfg
        geo
        n_mics
        hist_len               % history ring length in samples
    end

    properties (Access = private)
        hist_speech            % [hist_len x 1]
        hist_factory           % [hist_len x 1]
        hist_welding           % [hist_len x 1]
        ref_mic_idx
        frac_delays_speech_    % [7 x n_mics]
        frac_delays_factory_   % [7 x n_mics]
        frac_delays_welding_   % [7 x n_mics]
        gains_speech_          % [7 x n_mics]
        gains_factory_         % [7 x n_mics]
        gains_welding_         % [7 x n_mics]
    end

    methods
        function obj = SourceMixer(cfg, geo)
            obj.cfg    = cfg;
            obj.geo    = geo;
            obj.n_mics = cfg.n_mics;
            obj.ref_mic_idx = 1;

            % Cache delays and gains — read every block.
            obj.frac_delays_speech_  = obj.fetch_frac_delays_(geo, 'speech');
            obj.frac_delays_factory_ = obj.fetch_frac_delays_(geo, 'factory');
            obj.frac_delays_welding_ = obj.fetch_frac_delays_(geo, 'welding');
            
            obj.gains_speech_        = geo.gains_speech;
            obj.gains_factory_       = geo.gains_factory;
            obj.gains_welding_       = geo.gains_welding;

            tau_max = max([max(obj.frac_delays_speech_(:)), ...
                           max(obj.frac_delays_factory_(:)),  ...
                           max(obj.frac_delays_welding_(:)),    ...
                           0]);
            % Add one extra sample for the linear-interp right tap.
            obj.hist_len = ceil(tau_max) + 1;

            obj.hist_speech  = zeros(obj.hist_len, 1);
            obj.hist_factory = zeros(obj.hist_len, 1);
            obj.hist_welding = zeros(obj.hist_len, 1);
        end

        function mic = mix(obj, speech, factory, welding)
        %MIX  Run one block through the physical multi-source mixer.
            [speech, factory, welding, N] = obj.coerce_sources_(speech, factory, welding);

            sp_tape = [obj.hist_speech;  speech];
            fa_tape = [obj.hist_factory; factory];
            we_tape = [obj.hist_welding; welding];
            base    = obj.hist_len;                 % index of (sample-0-of-block - 1)

            mic = zeros(N, obj.n_mics);
            n_idx = (1:N).';
            
            for m = 1:obj.n_mics
                s_sum = zeros(N, 1);
                f_sum = zeros(N, 1);
                w_sum = zeros(N, 1);
                
                % Sum direct path and all 6 reflections
                for p = 1:7
                    s_q = (base + n_idx) - obj.frac_delays_speech_(p, m);
                    f_q = (base + n_idx) - obj.frac_delays_factory_(p, m);
                    w_q = (base + n_idx) - obj.frac_delays_welding_(p, m);

                    s_sum = s_sum + obj.gains_speech_(p, m)  * obj.frac_tap_(sp_tape, s_q);
                    f_sum = f_sum + obj.gains_factory_(p, m) * obj.frac_tap_(fa_tape, f_q);
                    w_sum = w_sum + obj.gains_welding_(p, m) * obj.frac_tap_(we_tape, w_q);
                end

                mic(:, m) = s_sum + f_sum + w_sum;
            end

            obj.hist_speech  = obj.push_(obj.hist_speech,  speech);
            obj.hist_factory = obj.push_(obj.hist_factory, factory);
            obj.hist_welding = obj.push_(obj.hist_welding, welding);
        end

        function reset(obj)
            obj.hist_speech(:)  = 0;
            obj.hist_factory(:) = 0;
            obj.hist_welding(:) = 0;
        end

        function ref = ref_mic(obj)
            ref = obj.ref_mic_idx;
        end

        function update_geometry(obj, new_geo)
        %UPDATE_GEOMETRY  Swap in a new geometry struct at runtime.
            if size(new_geo.gains_speech, 2) ~= obj.n_mics
                error('SourceMixer:update_geometry:NMicChange', ...
                      'Cannot change n_mics at runtime (have %d, got %d).', ...
                      obj.n_mics, size(new_geo.gains_speech, 2));
            end

            obj.geo                  = new_geo;
            obj.frac_delays_speech_  = obj.fetch_frac_delays_(new_geo, 'speech');
            obj.frac_delays_factory_ = obj.fetch_frac_delays_(new_geo, 'factory');
            obj.frac_delays_welding_ = obj.fetch_frac_delays_(new_geo, 'welding');
            
            obj.gains_speech_        = new_geo.gains_speech;
            obj.gains_factory_       = new_geo.gains_factory;
            obj.gains_welding_       = new_geo.gains_welding;

            tau_max      = max([max(obj.frac_delays_speech_(:)), ...
                                max(obj.frac_delays_factory_(:)),  ...
                                max(obj.frac_delays_welding_(:)),    0]);
            new_hist_len = max(1, ceil(tau_max) + 1);

            if new_hist_len > obj.hist_len
                pad = new_hist_len - obj.hist_len;
                obj.hist_speech  = [zeros(pad, 1); obj.hist_speech];
                obj.hist_factory = [zeros(pad, 1); obj.hist_factory];
                obj.hist_welding = [zeros(pad, 1); obj.hist_welding];
            elseif new_hist_len < obj.hist_len
                obj.hist_speech  = obj.hist_speech (end - new_hist_len + 1:end);
                obj.hist_factory = obj.hist_factory(end - new_hist_len + 1:end);
                obj.hist_welding = obj.hist_welding(end - new_hist_len + 1:end);
            end
            obj.hist_len = new_hist_len;
        end
    end

    methods (Access = private)
        function [speech, factory, welding, N] = coerce_sources_(~, speech, factory, welding)
            speech  = speech(:);
            factory = factory(:);
            welding = welding(:);
            N = max([numel(speech), numel(factory), numel(welding)]);
            if numel(speech)  < N, speech(end+1:N,1)  = 0; end
            if numel(factory) < N, factory(end+1:N,1) = 0; end
            if numel(welding) < N, welding(end+1:N,1) = 0; end
        end

        function y = frac_tap_(~, tape, q)
        %FRAC_TAP_  Linear-interpolated tap into a [history; current] tape.
            L  = numel(tape);
            q0 = floor(q);
            qf = q - q0;
            in = (q0 >= 1) & (q0 + 1 <= L);
            y  = zeros(numel(q), 1);
            idx_a = max(min(q0,     L), 1);
            idx_b = max(min(q0 + 1, L), 1);
            interp = (1 - qf) .* tape(idx_a) + qf .* tape(idx_b);
            y(in) = interp(in);
        end

        function h = push_(obj, h, src)
        %PUSH_  Slide the history ring left, append `src` on the right.
            N = numel(src);
            if N >= obj.hist_len
                h = src(end-obj.hist_len+1:end);
            else
                h = [h(N+1:end); src(:)];
            end
        end

        function tau = fetch_frac_delays_(~, geo, which)
        %FETCH_FRAC_DELAYS_  Read fractional delays for one source.
            name = ['frac_delays_' which];
            if isfield(geo, name)
                tau = geo.(name);
                return;
            end
            tau = double(geo.(['delays_' which]));
        end
    end
end
