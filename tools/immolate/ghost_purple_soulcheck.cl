// Filter for Immolate (https://github.com/SpectralPack/Immolate, CC BY-NC-SA 4.0); this file is shared under the same license.
// Ghost Deck, Purple Stake: tags, Arcana/Spectral draws (Soul -> joker + edition), shop Soul hits, packs, antes 1-3.
#define CACHE_SIZE 1000
#include "lib/immolate.cl"
long filter(instance* inst) {
    set_deck(inst, Ghost_Deck);
    set_stake(inst, Purple_Stake);
    init_locks(inst, 1, false, true);
    for (int ante = 1; ante <= 3; ante++) {
        init_unlocks(inst, ante, false);
        printf("\n==ANTE %i== boss ", ante); print_item(next_boss(inst, ante));
        printf(" | tags: "); print_item(next_tag(inst, ante)); printf(", "); print_item(next_tag(inst, ante));
        printf("\nFirst 4 Mega Arcana-size draws (ar1): \n");
        item cards[5];
        for (int p = 0; p < 4; p++) {
            arcana_pack(cards, 5, inst, ante);
            for (int i = 0; i < 5; i++) {
                print_item(cards[i]);
                if (cards[i] == The_Soul) { printf(" ("); item e = next_joker_edition(inst, S_Soul, ante); if (e != No_Edition) { print_item(e); printf(" ");} print_item(next_joker(inst, S_Soul, ante)); printf(")"); }
                printf(i<4?", ":"\n");
            }
        }
        printf("First 3 Mega Spectral-size draws: \n");
        for (int p = 0; p < 3; p++) {
            spectral_pack(cards, 5, inst, ante);
            for (int i = 0; i < 5; i++) { print_item(cards[i]); printf(i<4?", ":"\n"); }
        }
        printf("Shop items 1-60 containing Soul/legendary: ");
        for (int q = 1; q <= 60; q++) { shopitem it = next_shop_item(inst, ante); if (it.value == The_Soul) { printf("#%i The Soul ", q);} }
        printf("\nPacks 1-6: ");
        for (int q = 0; q < 6; q++) { print_item(next_pack(inst, ante)); printf("; "); }
        printf("\n");
    }
    return 0;
}
