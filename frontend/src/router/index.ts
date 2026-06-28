import {createRouter, createWebHistory} from 'vue-router';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {path: '/', redirect: '/datasets'},
    {path: '/datasets', component: () => import('../pages/Datasets.vue')},
    {path: '/experiments', component: () => import('../pages/Experiments.vue')},
    {path: '/inference', component: () => import('../pages/Inference.vue')},
  ],
});

export {router};
