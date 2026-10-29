<script setup>
import CharacterAvatar from './CharacterAvatar.vue'

defineProps({
  character: { type: Object, default: null },
})
</script>

<template>
  <div class="message-entry flex gap-3" role="status" aria-live="polite">
    <CharacterAvatar :name="character?.name ?? 'Терминал'" size="md" active />
    <div class="min-w-0 flex-1 space-y-2">
      <p class="flex items-center gap-3 text-[0.6rem] tracking-[0.25em] uppercase">
        <span class="font-semibold text-term-amber/80">{{ character?.name ?? 'терминал' }}</span>
        <span class="text-zinc-600">передача данных</span>
      </p>
      <div
        class="inline-flex items-center gap-2 border-l-2 border-term-amber/30 bg-zinc-900/40 px-3.5 py-2.5 text-sm text-term-green/80"
      >
        <span>Thinking</span>
        <span class="flex gap-1" aria-hidden="true">
          <span
            v-for="dot in 3"
            :key="dot"
            class="size-1 animate-pulse rounded-full bg-term-green/80"
            :style="{ animationDelay: `${dot * 0.15}s` }"
          />
        </span>
      </div>
    </div>
  </div>
</template>
