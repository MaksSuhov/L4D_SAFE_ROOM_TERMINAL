import { computed, ref } from 'vue'

import { describeError } from '@/api/errors'

/**
 * The roster of survivors together with the current selection.
 *
 * The first survivor returned by the API becomes the default responder, and a
 * selection is never kept across a reload if the character disappeared.
 */
export function useCharacters(api) {
  const characters = ref([])
  const selectedId = ref(null)
  const isLoading = ref(false)
  const error = ref(null)

  const selectedCharacter = computed(
    () => characters.value.find((character) => character.id === selectedId.value) ?? null,
  )

  async function load() {
    isLoading.value = true
    error.value = null

    try {
      const roster = await api.listCharacters()
      characters.value = roster

      const selectionIsValid = roster.some((character) => character.id === selectedId.value)
      if (!selectionIsValid) {
        selectedId.value = roster.at(0)?.id ?? null
      }
    } catch (cause) {
      characters.value = []
      selectedId.value = null
      error.value = describeError(cause, 'Не удалось получить список выживших.')
    } finally {
      isLoading.value = false
    }
  }

  function selectCharacter(characterId) {
    selectedId.value = characterId
  }

  return { characters, selectedId, selectedCharacter, isLoading, error, load, selectCharacter }
}
