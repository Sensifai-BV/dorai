function draw_scene(ax, geo, cfg)
%DRAW_SCENE  Render the static 3D acoustic scene into axes `ax`.
%   The view is centred on the human + robotic dog so they stay prominent.
%   Shows the room boundaries (floor, walls, ceiling) and both factory and
%   welding noise sources.
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

    margin = 0.6;

    % Bounds come from the human, dog, mic array
    key   = [geo.pos_human; geo.pos_dog; geo.pos_mics];
    focus = (geo.pos_human + geo.pos_dog) / 2;       % view centre

    rxy  = max(vecnorm(key(:,1:2) - focus(1:2), 2, 2));
    half = max(rxy + margin, 1.5);
    xr = focus(1) + [-half, half];
    yr = focus(2) + [-half, half];

    zr = [0, cfg.room_dims(3)];

    set(ax, 'DataAspectRatio', [1 1 1], 'PlotBoxAspectRatio', [1.1 1.1 0.9]);
    set(ax, 'XLim', xr, 'YLim', yr, 'ZLim', zr);

    % --- Draw Room (Closed Acoustic Space) ---
    L = cfg.room_dims(1);
    W = cfg.room_dims(2);
    H = cfg.room_dims(3);
    
    % Floor at z = 0
    [fx, fy] = meshgrid(linspace(-L/2, L/2, 10), linspace(-W/2, W/2, 10));
    surf(ax, fx, fy, zeros(size(fx)), ...
         'FaceColor', [0.22 0.22 0.25], 'FaceAlpha', 0.30, ...
         'EdgeColor', [0.40 0.40 0.45], 'EdgeAlpha', 0.15);
         
    % Ceiling at z = H
    surf(ax, fx, fy, H * ones(size(fx)), ...
         'FaceColor', [0.30 0.30 0.30], 'FaceAlpha', 0.05, ...
         'EdgeColor', [0.40 0.40 0.45], 'EdgeAlpha', 0.05);

    % Left Wall at x = -L/2
    [wy, wz] = meshgrid(linspace(-W/2, W/2, 10), linspace(0, H, 10));
    surf(ax, -L/2 * ones(size(wy)), wy, wz, ...
         'FaceColor', [0.18 0.18 0.20], 'FaceAlpha', 0.20, ...
         'EdgeColor', [0.35 0.35 0.40], 'EdgeAlpha', 0.10);
         
    % Back Wall at y = W/2
    [wx, wz] = meshgrid(linspace(-L/2, L/2, 10), linspace(0, H, 10));
    surf(ax, wx, W/2 * ones(size(wx)), wz, ...
         'FaceColor', [0.18 0.18 0.20], 'FaceAlpha', 0.20, ...
         'EdgeColor', [0.35 0.35 0.40], 'EdgeAlpha', 0.10);

    % --- Human body / mouth ---
    xh = geo.pos_human(1);  yh = geo.pos_human(2);
    Hh = cfg.human_height;
    plot3(ax, [xh xh xh], [yh yh yh], [0 Hh*0.52 Hh*0.87], ...
          'Color', [0.30 0.55 0.90], 'LineWidth', 2.5);
    th = linspace(0, 2*pi, 36);  rh = 0.08;
    plot3(ax, xh + rh*cos(th), yh + zeros(1,36), Hh*0.935 + rh*sin(th), ...
          'Color', [0.30 0.55 0.90], 'LineWidth', 2.0);
    scatter3(ax, xh, yh, geo.pos_human(3), 65, [0.30 0.60 1.0], 'filled', '^');
    text(xh+0.09, yh, geo.pos_human(3)+0.09, ...
         sprintf('Mouth %.2fm', geo.pos_human(3)), ...
         'FontSize', 7, 'Color', [0.40 0.70 1.0], 'Parent', ax);
    text(xh-0.08, yh, Hh+0.14, sprintf('%.0fcm', Hh*100), ...
         'FontSize', 7, 'FontWeight', 'bold', ...
         'Color', [0.40 0.60 0.90], 'Parent', ax);

    % --- Robotic Dog body + mic array ---
    draw_dog(ax, geo.pos_dog, geo.pos_mics, cfg);

    % Direct human -> dog path with distance label
    d = norm(geo.pos_dog - geo.pos_human);
    plot3(ax, [xh geo.pos_dog(1)], [yh geo.pos_dog(2)], ...
          [geo.pos_human(3) geo.pos_dog(3)], ...
          '--', 'Color', [0.30 0.60 1.0], 'LineWidth', 1.5);
    mid = (geo.pos_human + geo.pos_dog)/2;
    text(mid(1)+0.07, mid(2), mid(3)+0.10, sprintf('%.2f m', d), ...
         'FontSize', 8, 'FontWeight', 'bold', ...
         'Color', [0.55 0.80 1.0], 'Parent', ax);

    % --- Factory noise source ---
    pf = geo.pos_factory;
    scatter3(ax, pf(1), pf(2), pf(3), 85, [0.90 0.55 0.20], 's', 'filled');
    text(pf(1)+0.12, pf(2), pf(3)+0.12, 'Factory Noise', ...
         'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.95 0.65 0.30], 'Parent', ax);

    % --- Welding noise source ---
    pw = geo.pos_welding;
    scatter3(ax, pw(1), pw(2), pw(3), 85, [0.75 0.35 0.95], 'p', 'filled');
    text(pw(1)+0.12, pw(2), pw(3)+0.12, 'Welding Noise', ...
         'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.80 0.45 1.0], 'Parent', ax);

    xlabel(ax, 'X (m)'); ylabel(ax, 'Y (m)'); zlabel(ax, 'Z (m)');
    title(ax, 'Acoustic Scene (indoor / closed room)', ...
          'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
    view(ax, 38, 24);
    camproj(ax, 'perspective');
end
