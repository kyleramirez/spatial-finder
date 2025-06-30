import typescriptEslint from '@typescript-eslint/eslint-plugin';
import tsParser from '@typescript-eslint/parser';
import sveltePlugin from 'eslint-plugin-svelte';
import svelteConfig from 'eslint-plugin-svelte/lib/configs/flat/recommended.js';

export default [
  // Base
  {
    ignores: ['node_modules', 'dist', 'build'],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        projectService: {
          allowDefaultProject: ['eslint.config.js'],
        },
        extraFileExtensions: ['.svelte'],
      },
      globals: {
        browser: true,
        es2021: true,
        node: true,
      },
    },
    plugins: {
      '@typescript-eslint': typescriptEslint,
      svelte: sveltePlugin,
    },
    rules: {
      'no-multi-spaces': 'error',
      'no-unused-vars': ['error', { args: 'after-used', argsIgnorePattern: '^_' }],
      quotes: ['error', 'single', { avoidEscape: true, allowTemplateLiterals: true }],
      semi: ['error', 'always'],
    },
  },

  // Svelte
  ...svelteConfig,
  {
    files: ['**/*.svelte'],
    processor: sveltePlugin.processors.svelte,
    languageOptions: {
      parserOptions: {
        parser: tsParser,
        svelteConfig: {}, // Optional: add Svelte config if needed
      },
    },
    rules: {
      'svelte/no-at-html-tags': 'warn',
      'svelte/valid-compile': 'error',
    },
  },

  // TypeScript
  {
    files: ['**/*.ts', '**/*.tsx'],
    rules: {
      'arrow-parens': ['error', 'as-needed'],
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-implicit-any': 'error',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/explicit-module-boundary-types': 'off',
    },
  },
];
