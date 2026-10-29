<script setup>
import { Check, ChevronDown, UserRoundX } from 'lucide-vue-next'
import { computed, onBeforeUnmount, onMounted, ref, useTemplateRef } from 'vue'

import CharacterAvatar from './CharacterAvatar.vue'

const props = defineProps({
  characters: { type: Array, required: true },
  modelValue: { type: [Number, String], default: null },
  loading: { type: Boolean, default: false },
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue'])

const root = useTemplateRef('root')
const isOpen = ref(false)

const selected = computed(
  () => props.characters.find((character) => character.id === props.modelValue) ?? null,
)

const buttonLabel = computed(() => {
  if (props.loading) return 'Поиск выживших…'
  if (selected.value) return selected.value.name
  return props.characters.length === 0 ? 'Реестр пуст' : 'Выбрать выжившего'
})

function toggle() {
  if (props.disabled || props.loading) return
  isOpen.value = !isOpen.value
}

function choose(characterId) {
  emit('update:modelValue', characterId)
  isOpen.value = false
}

function closeOnOutsideClick(event) {
  if (root.value && !root.value.contains(event.target)) isOpen.value = false
}

function closeOnEscape(event) {
  if (event.key === 'Escape') isOpen.value = false
}

onMounted(() => {
  document.addEventListener('click', closeOnOutsideClick)
  document.addEventListener('keydown', closeOnEscape)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', closeOnOutsideClick)
  document.removeEventListener('keydown', closeOnEscape)
})
</script>

<template>
  <div ref="root" class="relative">
    <p class="mb-1 text-[0.55rem] tracking-[0.3em] text-zinc-600 uppercase">
      Responding agent
    </p>

    <button
      type="button"
      class="flex items-center gap-2 rounded-sm border border-zinc-700/70 bg-bunker-900/80 px-2.5 py-1.5 text-xs text-zinc-300 transition-colors hover:border-term-amber/40 hover:text-term-amber disabled:cursor-not-allowed disabled:text-zinc-600 disabled:hover:border-zinc-700/70"
      :disabled="disabled || loading"
      aria-haspopup="listbox"
      :aria-expanded="isOpen"
      @click="toggle"
    >
      <CharacterAvatar v-if="selected" :name="selected.name" size="sm" active />
      <UserRoundX v-else class="size-4 text-zinc-600" aria-hidden="true" />
      <span class="max-w-[9rem] truncate font-semibold">{{ buttonLabel }}</span>
      <ChevronDown
        class="size-3.5 transition-transform"
        :class="isOpen && 'rotate-180'"
        aria-hidden="true"
      />
    </button>

    <Transition name="pop">
      <ul
        v-if="isOpen"
        class="scrollbar-terminal absolute bottom-full left-0 z-30 mb-2 max-h-72 w-64 overflow-y-auto rounded-sm border border-term-amber/30 bg-bunker-900/95 shadow-[0_0_30px_-10px_rgba(255,176,0,0.35)] backdrop-blur"
        role="listbox"
        aria-label="Доступные выжившие"
      >
        <li class="border-b border-zinc-800/80 px-3 py-2 text-[0.55rem] tracking-[0.3em] text-zinc-600 uppercase">
          Выбрать канал связи
        </li>
        <li v-for="character in characters" :key="character.id">
          <button
            type="button"
            role="option"
            :aria-selected="character.id === modelValue"
            class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-xs transition-colors"
            :class="
              character.id === modelValue
                ? 'bg-term-amber/10 text-term-amber'
                : 'text-zinc-300 hover:bg-zinc-800/60'
            "
            @click="choose(character.id)"
          >
            <CharacterAvatar :name="character.name" size="sm" />
            <span class="truncate font-semibold">{{ character.name }}</span>
            <Check
              v-if="character.id === modelValue"
              class="ml-auto size-3.5 shrink-0"
              aria-hidden="true"
            />
          </button>
        </li>
      </ul>
    </Transition>
  </div>
</template>
