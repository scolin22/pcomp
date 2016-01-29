tbf = count3s_p3_time();
p = unique(sort(tbf(:,2)));
x = cell(length(p), 1);
leg = cell(length(p), 1);
s = zeros(length(p), 1);
for k = 1:length(p)
  ii = find(tbf(:,2) == p(k));
  x{k} = [ tbf(ii,1), tbf(ii,4) ./ tbf(ii,3) ];
end
palette = 'kbgrm';

ff = [figure; figure];
figure(ff(1)); hold('on');
for k = 1:length(p)
  tt = x{k};
  c = palette(mod(k-1, length(palette))+1);
  tail = (size(tt,1)-8):(size(tt,1));
  plot(tt(:,1), tt(:,2), [c, '-'], 'LineWidth', 2);
  s(k) = x{1}(end,2)/tt(end,2);
  if(k == 1)
    leg{k} = 'N_{proc}=1';
  else
    leg{k} = sprintf('N_{proc}=%d, speed-up=%4.1f', p(k), s(k));
  end
end
legend(leg, 'Location', 'NorthWest', 'FontSize', 16);

for k = 1:length(p)
  tt = x{k};
  c = palette(mod(k-1, length(palette))+1);
  figure(ff(1)); hold('on');
  plot(tt(tail,1), tt(tail,2), [c, '.'], 'MarkerSize', 25);
  figure(ff(2)); hold('on');
  head = 1:(size(tt,1)-7);
  plot(tt(head,1), tt(head,2), [c, '-'], 'LineWidth', 2);
  plot(tt(head((end-8):end),1), tt(head((end-8):end),2), [c, '.'], ...
  	'MarkerSize', 25);
end
figure(ff(1));
axis([0,1020000,0,0.016]);
ah = get(gcf, 'CurrentAxes');
set(ah, 'XTick', [0, 2, 4, 6, 8, 10]*100000, ...
    'XTickLabel', {'0', '200K', '400K', '600K', '800K', '1M'}, ...
    'FontSize', 16, 'YAxisLocation', 'right');
xh = xlabel('input size', 'FontSize', 20);
yh = ylabel('time (sec.)', 'FontSize', 20);
print('-depsc', 'bf_full');
% 
figure(ff(2)); hold('on');
ah = get(gcf, 'CurrentAxes');
set(ah, 'FontSize', 16);
set(ah, ...
  'XTick', (0:5)*10000, 'YTick', (0:8)*1.0e-4, ...
  'XTickLabel', {'0', '10K', '20K', '30K', '40K', '50K'}, ...
  'YTickLabel', {'0', '0.001', '0.002', '0.003', '0.004', '0.005', ...
  		 '0.006', '0.007', '0.008'});
  		 
xh = xlabel('input size', 'FontSize', 20);
yh = ylabel('time (sec)', 'FontSize', 20);
print('-depsc', 'bf_zoom');

b_1 = [ t_1(:,1), ones(size(t_1,1),1) ] \ (t_1(:,4) ./ t_1(:,3));
b_2 = [ t_2(:,1), ones(size(t_2,1),1) ] \ (t_2(:,4) ./ t_2(:,3));
