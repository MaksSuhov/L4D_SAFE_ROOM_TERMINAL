<script setup>
import { RadioTower, ShieldAlert } from 'lucide-vue-next'

defineProps({
  characterCount: { type: Number, default: 0 },
})

defineEmits(['suggestion'])

const SUGGESTIONS = [
  'Как убить Танка?',
  'Где искать аптечку в убежище?',
  'Что делать, если рядом Ведьма?',
  'Какое оружие брать против Орды?',
]
</script>

<template>
  <div class="flex h-full flex-col items-center justify-center gap-6 px-6 text-center">
    <pre
      class="text-[0.5rem] leading-tight text-term-amber/70 text-glow-amber select-none sm:text-[0.65rem]"
      aria-hidden="true"
    >  ____    _    _____ _____    ____   ___   ___  __  __
 / ___|  / \  |  ___| ____|  |  _ \ / _ \ / _ \|  \/  |
 \___ \ / _ \ | |_  |  _|    | |_) | | | | | | | |\/| |
  ___) / ___ \|  _| | |___   |  _ <| |_| | |_| | |  | |
 |____/_/   \_\_|   |_____|  |_| \_\\___/ \___/|_|  |_|</pre
    >

    <div class="space-y-2">
      <p class="flex items-center justify-center gap-2 text-xs tracking-[0.3em] text-term-green uppercase">
        <RadioTower class="size-4" aria-hidden="true" />
        Канал связи открыт
      </p>
      <p class="text-[0.7rem] tracking-[0.2em] text-zinc-400 uppercase">
        Выживших в реестре: {{ characterCount }}
      </p>
    </div>

    <p class="max-w-md text-sm leading-relaxed text-zinc-300">
      Спроси выжившего о заражённых, снаряжении или безопасном маршруте. Ответ приходит
      с опорой на архив лора убежища — источники печатаются под репликой.
    </p>

    <div class="flex flex-wrap justify-center gap-2">
      <button
        v-for="suggestion in SUGGESTIONS"
        :key="suggestion"
        type="button"
        class="rounded-sm border border-zinc-700/70 bg-bunker-900/60 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:border-term-amber/40 hover:text-term-amber"
        @click="$emit('suggestion', suggestion)"
      >
        {{ suggestion }}
      </button>
    </div>

    <p class="flex items-center gap-2 text-[0.6rem] tracking-[0.2em] text-zinc-600 uppercase">
      <ShieldAlert class="size-3.5" aria-hidden="true" />
      Не отходи от терминала без прикрытия
    </p>
  </div>
</template>
