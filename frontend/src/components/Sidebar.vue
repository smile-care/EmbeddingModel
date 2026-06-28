<script setup lang="ts">
import {onMounted, onUnmounted, ref} from 'vue';
import {RouterLink, useRoute} from 'vue-router';
import {
  Cpu,
  Database,
  FlaskConical,
  LogOut,
  Moon,
  Settings,
  Sun,
  ChevronRight,
  Terminal,
} from 'lucide-vue-next';
import {cn} from '@/lib/utils';
import {useTheme} from '@/composables/useTheme';

const route = useRoute();
const {theme, setTheme} = useTheme();

const settingsOpen = ref(false);
const settingsRef = ref<HTMLElement | null>(null);

function onDocMouseDown(e: MouseEvent) {
  if (!settingsOpen.value) return;
  if (settingsRef.value && !settingsRef.value.contains(e.target as Node)) {
    settingsOpen.value = false;
  }
}

onMounted(() => document.addEventListener('mousedown', onDocMouseDown));
onUnmounted(() => document.removeEventListener('mousedown', onDocMouseDown));

const navItems = [
  {path: '/datasets', label: '数据集', icon: Database},
  {path: '/experiments', label: '实验管理', icon: FlaskConical},
  {path: '/inference', label: '推理测试', icon: Terminal},
] as const;
</script>

<template>
  <aside class="flex h-screen w-64 flex-col border-r border-border bg-background">
    <div
      class="flex items-center gap-3.5 border-b border-border/80 px-6 py-7"
      aria-label="UnitX AI"
    >
      <div
        class="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 via-violet-600 to-purple-600 shadow-lg shadow-indigo-500/20 ring-1 ring-inset ring-white/15"
      >
        <Cpu class="h-[1.125rem] w-[1.125rem] text-white drop-shadow-sm" />
      </div>
      <div class="min-w-0 flex flex-col gap-0.5">
        <h1 class="flex flex-wrap items-baseline gap-x-1 text-[1.0625rem] font-semibold leading-none tracking-tight">
          <span class="text-foreground">UnitX</span>
          <span
            class="bg-gradient-to-r from-indigo-500 to-violet-500 bg-clip-text text-transparent dark:from-indigo-400 dark:to-violet-300"
          >
            AI
          </span>
        </h1>
        <p class="text-[10px] font-medium uppercase tracking-[0.2em] text-muted-foreground">
          Data Cluster
        </p>
      </div>
    </div>

    <nav class="flex-1 space-y-1 px-3">
      <RouterLink
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        custom
        v-slot="{href, navigate}"
      >
        <a
          :href="href"
          class="group flex items-center justify-between rounded-md px-3 py-2 text-sm font-medium transition-all"
          :class="
            cn(
              route.path === item.path
                ? 'bg-secondary text-foreground'
                : 'text-muted-foreground hover:bg-secondary/50 hover:text-foreground',
            )
          "
          @click="(e: MouseEvent) => navigate(e)"
        >
          <div class="flex items-center gap-3">
            <component :is="item.icon" class="h-4 w-4" />
            <span>{{ item.label }}</span>
          </div>
          <ChevronRight
            class="h-3 w-3 opacity-0 transition-transform group-hover:translate-x-1 group-hover:opacity-100"
          />
        </a>
      </RouterLink>
    </nav>

    <div class="mt-auto border-t border-border p-4">
      <div class="mb-2 flex items-center gap-3 px-2 py-3">
        <div class="h-8 w-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500" />
        <div class="flex min-w-0 flex-col overflow-hidden">
          <p class="truncate text-xs font-medium">Junhong Huang</p>
          <p class="truncate text-[10px] text-muted-foreground">junhong.huang@unitxlabs.com</p>
        </div>
      </div>
      <div class="flex flex-col gap-1">
        <div ref="settingsRef" class="relative">
          <button
            type="button"
            class="flex w-full items-center gap-3 rounded-md px-3 py-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-secondary/50 hover:text-foreground"
            :class="settingsOpen && 'bg-secondary/50 text-foreground'"
            @click="settingsOpen = !settingsOpen"
          >
            <Settings class="h-3.5 w-3.5" />
            Settings
          </button>

          <div
            v-show="settingsOpen"
            class="absolute bottom-full left-0 right-0 z-50 mb-2 rounded-lg border border-border bg-popover p-3 shadow-lg"
            @mousedown.stop
          >
            <p class="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              外观
            </p>
            <p class="mb-2 text-[11px] text-muted-foreground">主题</p>
            <div class="flex gap-1 rounded-md bg-secondary/50 p-1">
              <button
                type="button"
                class="flex flex-1 items-center justify-center gap-1.5 rounded px-2 py-1.5 text-[11px] font-medium transition-colors"
                :class="
                  theme === 'dark'
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                "
                @click="setTheme('dark')"
              >
                <Moon class="h-3 w-3" />
                深色
              </button>
              <button
                type="button"
                class="flex flex-1 items-center justify-center gap-1.5 rounded px-2 py-1.5 text-[11px] font-medium transition-colors"
                :class="
                  theme === 'light'
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                "
                @click="setTheme('light')"
              >
                <Sun class="h-3 w-3" />
                浅色
              </button>
            </div>
          </div>
        </div>
        <button
          type="button"
          class="flex items-center gap-3 rounded-md px-3 py-2 text-xs font-medium text-rose-500/80 transition-colors hover:bg-rose-500/10 hover:text-rose-500"
        >
          <LogOut class="h-3.5 w-3.5" />
          Logout
        </button>
      </div>
    </div>
  </aside>
</template>
