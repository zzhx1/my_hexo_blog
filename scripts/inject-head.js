/* Theme-independent <head> injections.
 * Kept here (not in theme config) so they survive theme switches. */

/* Google Search Console verification meta — originally set in Butterfly's site_verification. */
hexo.extend.injector.register(
  'head_end',
  '<meta name="google-site-verification" content="mspInednFW1hhgphA_wKNGPrFamhQAiJk9IWXaWsT7Y">',
  'default'
);
