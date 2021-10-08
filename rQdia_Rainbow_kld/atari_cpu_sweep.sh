#!/bin/sh
module load python3/3.8.3
seed=3
prefix="kld"
#for game in "alien" "amidar" "assault" "asterix" "bank_heist" "battle_zone" "boxing" "breakout" "chopper_command" "crazy_climber" "demon_attack" "freeway" "frostbite" "gopher" "hero" "jamesbond" "kangaroo" "krull" "kung_fu_master" "ms_pacman" "pong" "private_eye" "qbert" "road_runner" "seaquest" "up_n_down"
#for game in "alien" "amidar" "assault" "asterix" "bank_heist" "boxing" "breakout" "chopper_command"
for game in "breakout" "battle_zone"
do
	python3 sbatch.py --cpu --name rerun2$prefix$game$seed --params "--game $game --seed $seed --expname rerun2$prefix$game$seed"
	sleep 2
done
