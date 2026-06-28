import {inject, provide, ref, watch, type InjectionKey, type Ref} from 'vue';

export type Theme = 'dark' | 'light';

const STORAGE_KEY = 'unitx-theme';

export type ThemeContextValue = {
  theme: Ref<Theme>;
  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
};

const themeKey: InjectionKey<ThemeContextValue> = Symbol('theme');

function readStoredTheme(): Theme {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === 'light' || v === 'dark') return v;
  } catch {
    /* ignore */
  }
  return 'dark';
}

export function provideTheme() {
  const theme = ref<Theme>(typeof document !== 'undefined' ? readStoredTheme() : 'dark');

  watch(
    theme,
    (t) => {
      const root = document.documentElement;
      if (t === 'light') {
        root.classList.add('light');
      } else {
        root.classList.remove('light');
      }
      try {
        localStorage.setItem(STORAGE_KEY, t);
      } catch {
        /* ignore */
      }
    },
    {immediate: true},
  );

  const setTheme = (t: Theme) => {
    theme.value = t;
  };

  const toggleTheme = () => {
    theme.value = theme.value === 'dark' ? 'light' : 'dark';
  };

  const value: ThemeContextValue = {
    theme,
    setTheme,
    toggleTheme,
  };

  provide(themeKey, value);
  return value;
}

export function useTheme(): ThemeContextValue {
  const ctx = inject(themeKey);
  if (!ctx) {
    throw new Error('useTheme must be used within App (ThemeProvider)');
  }
  return {
    theme: ctx.theme,
    setTheme: ctx.setTheme,
    toggleTheme: ctx.toggleTheme,
  };
}
