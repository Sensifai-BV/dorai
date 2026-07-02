function print_geometry(geo)
%PRINT_GEOMETRY  Pretty-print the scene geometry returned by build_geometry().
    fprintf('=== Geometry (indoor / closed room) ===\n');
    fprintf('  Mouth     : [%.3f %.3f %.3f] m\n', geo.pos_human);
    fprintf('  Robotic Dog: [%.3f %.3f %.3f] m\n', geo.pos_dog);
    fprintf('  Factory N : [%.3f %.3f %.3f] m\n', geo.pos_factory);
    fprintf('  Welding N : [%.3f %.3f %.3f] m\n', geo.pos_welding);
    for m = 1:size(geo.pos_mics,1)
        fprintf('  Mic%d pos=[%.3f %.3f %.3f]  d=%.3f m  TDOA=%d smp\n', ...
            m, geo.pos_mics(m,1), geo.pos_mics(m,2), geo.pos_mics(m,3), ...
            geo.dist_speech(m), geo.delays(m));
    end
    fprintf('\n');
end
