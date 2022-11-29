from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.bot_ai import BotAI
import numpy as np
import get_features

class WorkerRushBot(BotAI):
    async def on_step(self, iteration: int):
        if iteration == 0:
            for worker in self.workers:
                worker.attack(self.enemy_start_locations[0])

class SimpleMineralBot(BotAI):
    async def on_start(self):
        self.client.game_step: int = 2
        
    async def on_step(self, iteration: int):

        for worker in self.workers:
            meele_enemies = self.enemy_units.closer_than(3, worker)
            #meele_enemies = list(filter(worker.target_in_range, list(self.enemy_units)))
            if worker.shield > 0 and len(meele_enemies) > 0:
                meele_enemies.sort(key=worker.distance_to)
                worker.attack(meele_enemies[0])
            elif not worker.is_collecting:
                field = self.mineral_field.random
                worker.gather(field)


    def get_features_1(self):
        for worker in self.workers:
            self.get_indiv_features_1(worker)

    def get_indiv_features_1(self, unit):
        '''
        All enemy/allies distances, health, shield, self health, shield, has minerals
        '''
        enemies = self.enemy_units.sorted_by_distance_to(unit)
        allies = self.units.sorted_by_distance_to(unit)
        enemy_feats = []
        ally_feats = []
        self_feats = []
        for enemy in enemies:
            enemy_feats.append(enemy.distance_to_squared(unit),enemy.health)
        for ally in allies:
            ally_feats.append(ally.distance_to_squared(unit), ally.health, ally.shield)

        self_feats.append(unit.health, unit.shield, int(unit.is_carrying_resource))

        return enemy_feats + ally_feats + self_feats
        

        



def main():
    #run_game(maps.get("Abyssal Reef LE"), [
    run_game(maps.get("MineAndKillZerglings"), [Bot(Race.Protoss, SimpleMineralBot())], realtime=True)

if __name__ == '__main__':
    main()