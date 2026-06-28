<script setup lang="ts">
import {RouterView} from 'vue-router';
import Sidebar from './components/Sidebar.vue';
import {provideTheme} from './composables/useTheme';

provideTheme();
</script>

<template>
  <div class="flex h-screen w-full overflow-hidden bg-background text-foreground">
    <Sidebar />
    <main class="flex-1 overflow-y-auto">
      <RouterView />
    </main>
  </div>
</template>
