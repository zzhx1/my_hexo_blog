/* 生成一个空的 .nojekyll 到产物根目录，让 GitHub Pages 跳过 Jekyll 处理。
 * 背景：Pages 默认用 Jekyll 处理静态文件，而 Hexo/Fluid 的产物会让 Jekyll
 * 构建失败（conclusion=failure），导致线上一直回退到旧版本。有了 .nojekyll，
 * Pages 直接当纯静态站点发布。每次 hexo generate 自动生成，无需额外依赖。 */
hexo.extend.generator.register('nojekyll', function () {
  return { path: '.nojekyll', data: '' };
});
