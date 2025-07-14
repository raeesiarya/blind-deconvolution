<script setup lang="ts">
import { loginWithMicrosoft } from './auth'
import axios from 'axios'

const login = async () => {
  const token = await loginWithMicrosoft()
  await axios.post('http://localhost:8000/api/auth/', { token })
}
</script>

<template>
  <button @click="login">Login with Microsoft</button>
</template>
