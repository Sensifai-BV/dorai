function draw_dog(ax, pd, pm, cfg)
%DRAW_DOG  Draw 3D robotic dog (artaban robot from panza) in the scene axes.
%   pd: dog position [1x3] (center of the back, height 0.50m)
%   pm: mic positions [N x 3] on its back
%   cfg: configuration struct

    dx = pd(1); dy = pd(2); dz = pd(3); % dz = 0.50m
    
    % Torso box dimensions (Artaban robotic dog body)
    len = 0.45; wd = 0.20; ht = 0.15;
    z_top = dz;
    z_bot = dz - ht;
    
    % 8 vertices of the torso box
    V = [dx-len/2 dy-wd/2 z_bot; ... % 1
         dx+len/2 dy-wd/2 z_bot; ... % 2
         dx+len/2 dy+wd/2 z_bot; ... % 3
         dx-len/2 dy+wd/2 z_bot; ... % 4
         dx-len/2 dy-wd/2 z_top; ... % 5
         dx+len/2 dy-wd/2 z_top; ... % 6
         dx+len/2 dy+wd/2 z_top; ... % 7
         dx-len/2 dy+wd/2 z_top];    % 8
    
    F = [1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 2 3 7 6; 1 4 8 5];
    
    % Torso patch (Sleek metallic dark gray body for Artaban)
    patch('Parent', ax, 'Vertices', V, 'Faces', F, ...
          'FaceColor', [0.22 0.22 0.25], 'FaceAlpha', 0.90, ...
          'EdgeColor', [0.40 0.40 0.45], 'LineWidth', 1.0);
          
    % 4 legs (from torso bottom corners to the ground)
    leg_offset_x = len/2 - 0.05;
    leg_offset_y = wd/2 - 0.02;
    leg_xs = dx + [leg_offset_x, leg_offset_x, -leg_offset_x, -leg_offset_x];
    leg_ys = dy + [leg_offset_y, -leg_offset_y, leg_offset_y, -leg_offset_y];
    
    for i = 1:4
        % Leg from torso bottom to ground
        plot3(ax, [leg_xs(i) leg_xs(i)], [leg_ys(i) leg_ys(i)], [z_bot 0], ...
              'Color', [0.30 0.30 0.35], 'LineWidth', 3.0);
        % Foot marker on floor
        scatter3(ax, leg_xs(i), leg_ys(i), 0, 15, [0.15 0.15 0.15], 'filled');
    end
    
    % Head box (on the front side facing the human at the origin)
    to_human = -[dx, dy];
    to_human = to_human / (norm(to_human) + 1e-9);
    
    head_size = 0.11;
    head_center = [dx + to_human(1)*(len/2 + 0.03), dy + to_human(2)*(len/2 + 0.03), dz + 0.06];
    
    V_head = [head_center(1)-head_size/2 head_center(2)-head_size/2 head_center(3)-head_size/2; ...
              head_center(1)+head_size/2 head_center(2)-head_size/2 head_center(3)-head_size/2; ...
              head_center(1)+head_size/2 head_center(2)+head_size/2 head_center(3)-head_size/2; ...
              head_center(1)-head_size/2 head_center(2)+head_size/2 head_center(3)-head_size/2; ...
              head_center(1)-head_size/2 head_center(2)-head_size/2 head_center(3)+head_size/2; ...
              head_center(1)+head_size/2 head_center(2)-head_size/2 head_center(3)+head_size/2; ...
              head_center(1)+head_size/2 head_center(2)+head_size/2 head_center(3)+head_size/2; ...
              head_center(1)-head_size/2 head_center(2)+head_size/2 head_center(3)+head_size/2];
              
    patch('Parent', ax, 'Vertices', V_head, 'Faces', F, ...
          'FaceColor', [0.28 0.28 0.32], 'FaceAlpha', 0.90, ...
          'EdgeColor', [0.45 0.45 0.50], 'LineWidth', 1.0);
          
    % Draw ears (Artaban robotic dog details)
    plot3(ax, [head_center(1) head_center(1)], ...
              [head_center(2)-head_size/2 head_center(2)-head_size/2-0.02], ...
              [head_center(3)+head_size/2 head_center(3)+head_size/2+0.04], ...
              'Color', [0.22 0.22 0.25], 'LineWidth', 2.0);
    plot3(ax, [head_center(1) head_center(1)], ...
              [head_center(2)+head_size/2 head_center(2)+head_size/2+0.02], ...
              [head_center(3)+head_size/2 head_center(3)+head_size/2+0.04], ...
              'Color', [0.22 0.22 0.25], 'LineWidth', 2.0);

    % Plot mic array on the dog's back
    mc = {[0.00 0.78 0.32], [0.00 0.55 0.95], [1.00 0.52 0.00], ...
          [0.95 0.35 0.75], [0.60 0.85 0.20]};
    for m = 1:size(pm, 1)
        idx = mod(m-1, numel(mc)) + 1;
        scatter3(ax, pm(m,1), pm(m,2), pm(m,3), 60, mc{idx}, 'filled', 'o');
        text(pm(m,1), pm(m,2), pm(m,3) + 0.055, ...
             sprintf('M%d', m), 'FontSize', 7, 'Color', mc{idx}, 'Parent', ax);
    end

    % Text label for the robotic dog
    text(dx + 0.12, dy, dz + 0.10, ...
         'Artaban (Dog) H=50cm', ...
         'FontSize', 8, 'FontWeight', 'bold', ...
         'Color', [0.95 0.75 0.40], 'Parent', ax);
end
