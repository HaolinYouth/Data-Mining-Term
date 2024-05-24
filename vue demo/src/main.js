import { createApp } from 'vue'
import App from './App.vue'
import './registerServiceWorker'

import axios from 'axios'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

const app = createApp(App)

app.config.globalProperties.$axios = axios

app.use(ElementPlus)
app.mount('#app')
