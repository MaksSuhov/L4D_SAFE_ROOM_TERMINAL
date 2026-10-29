<script setup>
import { computed, ref, watch } from 'vue'

import { personaAccent, personaInitials, personaPortrait } from '@/lib/personas'

const props = defineProps({
  name: { type: String, required: true },
  /** Optional picture supplied by the API; the shipped portrait is the default. */
  avatarUrl: { type: String, default: null },
  size: { type: String, default: 'md' },
  active: { type: Boolean, default: false },
})

const SIZES = {
  sm: 'size-7 text-[0.6rem]',
  md: 'size-9 text-xs',
  lg: 'size-12 text-sm',
}

const accent = computed(() => personaAccent(props.name))
const initials = computed(() => personaInitials(props.name))

const picture = computed(() => props.avatarUrl || personaPortrait(props.name))

/** A missing file must not leave a broken image behind — the monogram returns. */
const hasFailed = ref(false)
watch(picture, () => {
  hasFailed.value = false
})

const style = computed(() => ({
  borderColor: accent.value.border,
  backgroundColor: accent.value.background,
  color: accent.value.accent,
  boxShadow: props.active ? `0 0 14px -2px ${accent.value.glow}` : 'none',
}))
</script>

<template>
  <span
    class="inline-flex shrink-0 select-none items-center justify-center overflow-hidden rounded-sm border font-bold tracking-wider uppercase"
    :class="SIZES[size] ?? SIZES.md"
    :style="style"
    :title="name"
  >
    <img
      v-if="picture && !hasFailed"
      :src="picture"
      :alt="name"
      class="size-full object-cover"
      @error="hasFailed = true"
    />
    <template v-else>{{ initials }}</template>
  </span>
</template>
