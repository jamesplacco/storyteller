# Storyteller experiment report: first-run

Generated 2026-11-11T16:24:49 by `report.py` at commit `fa302ea`. Every number below was measured on this machine; every sample uses a fixed prompt and seed, and none were picked by hand.

## Setup

| | |
|---|---|
| platform | macOS-26.6.2-arm64-arm-64bit |
| machine | arm64 |
| python | 3.9.6 |
| device | mps |
| chip | Apple M3 |
| memory_gb | 8 |
| libraries | torch 2.8.0, numpy 2.0.2, datasets 4.5.0, fastapi 0.128.8, PIL 11.3.0 |
| train tokens | 120,511,848 |
| val tokens | 4,825,773 |
| sft pairs | 15,660 |

## Models

| checkpoint | parameters | MB | training recorded in the checkpoint |
|---|---:|---:|---|
| `ckpt.pt` | 13,891,584 | 49.3 | script train.py, steps_done 5,000, max_steps 5,000, tokens_seen 40,960,000, batch_size 32, block_size 256, grad_accum 1, dtype fp32, max_lr 0.001, device mps, world_size 1, seconds 3722.2, val_loss 1.8505 |
| `chat.pt` | 13,891,584 | 49.3 | script sft.py, from ckpt.pt, steps 1,500, seconds 465.0 |
| `chat_dpo.pt` | 13,891,584 | 49.3 | script dpo.py, from chat.pt, steps 300, seconds 116.8 |
| `chat_ppo.pt` | 13,891,584 | 49.3 | script ppo.py, from chat.pt, iters 150, seconds 1000.8 |
| `reward.pt` | 13,891,969 | 49.3 | not recorded |
| `vqvae.pt` | 1,308,739 | 5.24 | not recorded |
| `image_gpt.pt` | 11,073,408 | 43.52 | not recorded |

SHA-256 checksums for every file are in `manifest.json`.

## Evaluation

```
checkpoint     perplexity  topic word  finished  repeats  length
ckpt.pt              6.49                                          (base model: not trained to follow requests)
chat.pt              6.59         36%       80%       1%     174
chat_dpo.pt          6.65         96%       70%       2%     178
chat_ppo.pt          6.68         44%       98%       1%     162
50 prompts per chat model, same prompts and seeds for each.
```

- **perplexity**: on held-out validation stories. Lower is better.
- **topic word**: share of stories containing the requested word. A word match only, not a judgment of quality.
- **finished**: share of stories the model ended itself. **repeats**: share of 4-word phrases that repeat earlier ones.
- 50 prompts per chat model, phrasings not used in training, the same prompts and seeds for every model.

## Sample stories

### Base model (`ckpt.pt`), prompt "Once upon a time"

**greedy:** Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a big, scary dog running towards her. The dog barked and ran towards her.

Lily was scared and ran away. She ran as fast as she could, but the dog was faster. Lily was scared and didn't know what to do. She ran back to her house and told her mom what happened. Her mom hugged her and said, "Don't worry, Lily. We can go back to the park and play with your ball."

Lily was happy and went back to playing with her ball. She played with her ball and had lots of fun. When it was time to go home, Lily said goodbye to the dog and went back to her house. She was happy to have a new friend and to have a fun day at the park.

**seed 0:** Once upon a time, there was a little boy named Timmy. He loved to play outside with his friends. One day, Timmy and his friends found a big, shiny rock in the park. They were all so happy and excited to show it to their friends.

But then, a big naughty dog came and chased the rock away. Timmy and his friends cheered and laughed. They decided to hide the rock, but they didn't know how to use it.

Finally, a hero came and gave it back to the boy. Timmy was so happy and grateful for the hero. He realized that the rock was an even more important than to play with. From that day on, Timmy learned that it's not boring after all.

**seed 1:** Once upon a time, there was a little girl. She loved to explore and have fun. One day she decided to go for a walk in the woods. As she was walking, she heard a voice. It was coming from a small house. The little girl walked over to it and saw a friendly old man with a big, white beard and a big, white beard.

The old man smiled and said, "Hello, little girl. Do you want to come with me?" 

The little girl was so excited! She ran off to explore the woods. Suddenly, she heard a voice behind her. "Hello, little girl. What are you doing here?"

The old man replied, "I'm here to deliver this place. I have a lot of fun - please!" 

The little girl was determined to help. She went up to the old man and gave him the case.

**seed 2:** Once upon a time, there was a little bird who lived in a big tree. One day, the bird fell down and hurt his leg. He felt like crying.

The bird called out to his friends and tried to help him. They decided to help him find his leg. They searched high and low,, and finally found his leg still hurt.

The bird was happy again and felt better. He flew around the tree, trying to get his leg back. He took his leg out and felt very good, as he was able to fly.

The bird's friends cheered him on, and they were all very happy. They thanked the bird and flew off to find his leg. And they all lived happily ever after.

### Prompt 1: "Could you tell me a short tale about a moon?" (seed 0)

**`chat.pt`** (topic word missing, finished): Once upon a time, there was a little girl named Lily. She loved to play outside in the forest and explore new things. One day, she saw a beautiful purple butterfly. She wanted to catch it, but her mother said no. Lily was sad and didn't know what to do. 

Then, a kind witch appeared and asked her if she could help. The witch said, "Yes, I can help you catch the butterfly." The witch tried to pull the butterfly, but it didn't work. Suddenly, Lily's mother came to the forest and said, "Lily, I know how to catch the butterfly. Maybe you can give it some honey." 

Lily was amazed and started to feed the butterfly. She caught it in her hands and it flew away. She was very happy and felt proud of herself for being brave. From that day on, she and the witch were the best of friends.

**`chat_dpo.pt`** (topic word present, finished): Once there was a little boy. He was three years old and he loved the moon. Every day he asked his mom to get him a special moon and if he could take him with him.

One day, he was at the moon and he asked her if he could go to the moon. She loved the moon very much and they loved it.

The little boy said yes and he brought his toy a big smile. The moon was bright and he could not believe his eyes. He told the little boy how much fun it was to go there.

The little boy and the moon went back to the moon every day. They would play and laugh together until it was time to go home.

The little boy kept exploring every night, until one day he found a big, blue moon. He felt so excited that he ran back to the moon to tell his mom.

The little boy waved his toy moon, and he knew that the moon was very light.

**`chat_ppo.pt`** (topic word present, cut off): Once upon a time, there was a little girl named Lily. She loved to play outside in the forest. One day, she saw a beautiful moon and picked it up to see it. She held the moon in her hand and started to fly in the air. She felt so happy and free. 

Suddenly, a big gust of wind blew Lily and she got very dizzy. She didn't like flying or twirling up. She started to shiver and didn't know why it was so scary. 

Her mommy saw her below and said, "Don't worry, Lily. I'll help you. Just protect Lily and put her in your pocket." Lily followed her mommy to a nearby forest where they were safe. 

As they were walking, Lily felt a gentle wind around her face. She didn't want to get too dizzy, but her mommy encouraged her to hold on to keep her breath. They flew around and Lily felt safe and warm.

### Prompt 2: "Could you tell me a short tale about a cat?" (seed 1)

**`chat.pt`** (topic word missing, cut off): Once upon a time, there was a little mouse who lived in a big house. He was very happy living in his cozy house, but he was always too sad for anyone to play with. One day, he decided to go for a walk in the woods. He saw a big tree and he decided to try and climb it. 

As he was climbing, he saw a huge pile of dirt with little rocks. He was so excited that he wanted to climb it. But when he got to the top of the pile, he noticed that it was too deep and his fingers were bumpy. 

He heard a voice calling out, â€œWhoâ€™s there?â€ It was a tiny voice.

The little mouse opened his eyes and saw a friendly friendly old woman with a white beard. â€œHello! Would you like to come out for a walk with me?â€ she asked.

The old woman smiled and said yes, and the little mouse was so excited to come along the

**`chat_dpo.pt`** (topic word present, finished): Once upon a time, there was a cat. The cat had a big, blue tail. The cat liked to play with its tail all day. One day, the cat saw a big cat. The cat wanted to play with the cat.

The cat said, "Hi, cat. Do you want to play?" The cat said, "Yes, let's play." The cat and the cat played and had fun. The cat and the cat were very happy.

After playing, the cat and the cat became good friends. They played together every day at the same game. The cat and the cat had many fun in the big, red field. They were never hungry or happy.

**`chat_ppo.pt`** (topic word present, finished): Once upon a time, there was a little mouse named Timmy. One day, Timmy was playing in the kitchen. He saw a big jar of candy on the table. He wanted to eat all of it, but it was too heavy for him. 

Timmy asked his mom for help. "Can you help me open the jar?" he asked. His mom said, "Sure, you can take the jar." Timmy carefully lifted the jar up and tried to put the candy inside. 

Suddenly, a big cat came to Timmy and said, "Hello, little mouse! Do you want to open the jar too?" Timmy was scared but he trusted his mom. So they opened the jar together and saw a little mouse stuck inside. Timmy was so grateful. He thanked his mom and hugged him.

### Prompt 3: "Could you tell me a short tale about a kite?" (seed 2)

**`chat.pt`** (topic word missing, finished): One day, two friends, Jack and their mother were walking down the street. They were looking for something special to eat. Suddenly, they saw a small bird outside the window.

"Look, Mommy! The bird is so pretty!" Jack said.

The mother smiled. "Do you think the bird needs some food?"

Jack felt a little scared. He nodded and said, "Yes, I think the bird is hungry!"

Dad smiled. "Let's go get some food, let's go home," he said.

Jack and his mother walked back, handing the bird to eat. It was a big, furry poppy and Jack was very happy.

**`chat_dpo.pt`** (topic word present, finished): Once upon a time, there was a big kite. It was so happy to fly high in the sky. The kite was so excited that it could jump up and down and fly high!

The kite started to fly very high and flew higher and higher. It was so high up that it was going so fast! But still the kite felt so happy and flying.

But then, the kite began to get tired and couldn't fly anymore. It was so tired, but then it was so happy it started to dance too.

The kite's kite flew so high, it could not hear it anymore. It was flying high and far away from the wind.

The kite was so happy that it flew all the way back into the sky and back to its kite. It had made it so much fun!

**`chat_ppo.pt`** (topic word missing, finished): Once upon a time, there was a big garden. In the garden lived a little bird. The bird liked to fly high in the sky. One day, a big storm came. The wind was very strong and the bird was scared. 

The bird flew away, but the bird did not go with a plan. The wind blew harder and the rain started to pour. The bird was so scared that it kept flying. 

But then, a big wind blew hard and the bird felt very strong. It was very high! The bird could not fly anymore. The bird was sad, but it was still not sure how to help the bird. 

The bird and the bird became friends. They would race together in the garden every day. The nest was always very happy and safe. The bird was happy, because it had a friend to help, even when the storm was hard.

### Prompt 4: "Could you tell me a short tale about a moon?" (seed 3)

**`chat.pt`** (topic word present, finished): Once upon a time, there was a girl who loved to spin. She spun around in circles, spinning in circles, and even her head on the ground. One day, she wanted to make a wish, so she put on her special shoes and started to spin. She spun faster and faster, until she saw the moon. She jumped and spun until she was the moon! She spun around and around, spinning and feeling the moon ray of energy. When she was done, she ran back into the sun, spinning and twirling with joy. The world around her was so happy that she was able to make the world a better place.

**`chat_dpo.pt`** (topic word present, finished): Once upon a time, there was a bright orange moon. The moon shone brightly and sparkled in the sun. Byan was so excited as he explored it.

Suddenly, the moon disappeared! Byan looked up and saw a beautiful fairy flying in the sky. She waved with a smile.

The fairy flew around in the sky, and then a big wave came and made her to the moon. Byan was so happy and excited.

But then, the moon started to get too close. Byan was so sad. The fairy waved goodbye and flew away.

Then, Byan saw the moon. She waved goodbye and the fairy flew away. Byan waved goodbye and ran back to the moon, feeling sad and alone.

Byan thought about the magical moon. He waved goodbye and went back to his distant moon, eager to explore it.

**`chat_ppo.pt`** (topic word missing, finished): Once upon a time, there was a lion who was very happy. He loved to clap around in the jungle. One day, he met a little bunny who was feeling very embarrassed. He asked the rabbit, "What's wrong doing, little bunny?"

The bunny replied, "My favorite toy is a teddy bear. I can't clap myself."

The bunny smiled and said, "Don't worry, I'll clap your hands together!"

The bear clapped his hands together and soon he was laughing. The bunny was having so much fun that he started to clap too. He clapped his hands together and they both enjoyed the big clapped.

He was the happiest animal in the jungle for his animal friends to join. So the bunny clapped his hands together and even danced back to the forest.

### Prompt 5: "Could you tell me a short tale about a train?" (seed 4)

**`chat.pt`** (topic word present, finished): Once upon a time, there was a poor man. He was sad because he couldn't get married. So he decided to go to the park. He put on his shoes and ran to the park. When he got there, he spotted something very special. It was a big, blue train. The man was so excited that he wanted to take it home.

He ran away as fast as he could and soon he reached the station. He was so happy when he saw the man. He wanted to bring it home.

He walked inside the train and saw the station. The man called out to him: "Do not go there!" He ran back home and took off his shoes.

The man kept walking and soon came to the station. He was so happy to be home. He was so glad that he had been able to get the train.

**`chat_dpo.pt`** (topic word present, finished): Once upon a time, there was a powerful engine. The engine was brave and strong, so he went on a train ride. The engine was mighty, with lots of lights, cars and people who saw the most amazing things. 

One day, the engine stopped and saw something very interesting in the station. In the station was a huge train and they were going to take it for a ride. The engine was excited and hopped into the train with it in his hands. 

The engine rode in the engine all the way to the station, but, the engine never stopped. The station was a small train, and they were all on the way. 

The engine had saved and the driver was very happy. He knew he had been too strong, and he could be very strong. 

The engine rode off, feeling like he was able to move on a powerful train. And he was ready to ride the train.

**`chat_ppo.pt`** (topic word missing, finished): Once upon a time, there was a little girl named Lily. She loved to ride her bike around the car. One day, she saw that her dad had a big surprise waiting for her. Lily was so happy and wanted to celebrate with her new friend.

When Lily arrived at the park, she saw a toy she really wanted, but she had no money. She was sad because she didn't wear her bike. But then, her mom told her it was okay to take care of her helmet.

Lily took a deep breath and walked around the park. She learned how to ride her bike, and how to ride on it. She felt great without a feeling and took off her helmet. It was a fun day for Lily to ride her bike.

### Prompt 6: "Please write a story where a princess is the hero." (seed 5)

**`chat.pt`** (topic word missing, cut off): Once upon a time, there was a little boy named Timmy. Timmy liked to play with his toys and eat yummy food. One day, Timmy's mommy told him they needed to go to the store to buy some ice cream. Timmy was very excited to go.

When they got to the store, Timmy saw a big machine that was moving around. He wanted to see what was at the machine too. So, Timmy asked his mommy if he could have the special treat. His mommy said no and told him to be careful.

After they left the store, Timmy found a pretty flower that he liked to smell. He showed it to his mommy and daddy. They thought it was the best ice cream ever. Timmy was so happy and he wanted a piece of the ice cream.

But when they got home, Timmy's room was empty. He was sad because he really wanted the flower and he didn't want it anymore. Timmy's mommy and daddy took the flower

**`chat_dpo.pt`** (topic word present, cut off): Once upon a time, there was a little boy named Lily. Lily had a furry costume and loved to play with her toys. One day, Lily's mom took her to a big castle. Lily was so excited to see all the other princess toys, but she was scared to go inside. 

Lily's mom took her to a party and Lily saw a scary monster wearing a beautiful dress. She asked her mom, "What's that big, scary monster?" Her mom replied, "It's just a girl who ran away from my kingdom." 

Lily felt scared and didn't want to leave the weird monster. She wanted to go in the castle, but she was worried it would be bad. But her mom had an idea and asked Lily to go in the castle. They looked inside and saw a big, scary monster. 

Lily was scared and tried to run. But the monster was strong and she was brave enough to go out. Lily was scared and

**`chat_ppo.pt`** (topic word missing, finished): Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toys and eat cookies. One day, Timmy's mom asked him to make a cake with some dough. Timmy was excited to help his mom mix a cake into the cake.

After they finished mixing, Timmy's dad asked him to help cook a cake. Timmy started to mix the mix, but he accidentally dropped the dough on the floor. His dad told him it was dangerous. Timmy felt bad and said sorry to his dad.

After they made the cake together, Timmy felt better knowing he was not hurt anymore. From then on, Timmy always made sure to make sure to always check if it was still in the kitchen.

All 50 stories per model are in `eval.json`.

## Illustrations

The app picks one of CIFAR-10's 10 classes from words in the story and draws a 32×32 picture of that class. It is conditioned on the class only, not on the story's details.

![Image GPT (Chapter 17): 6 samples per class, seed 0. Rows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.](image_gpt.png)

Image GPT (Chapter 17): 6 samples per class, seed 0. Rows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Systems, measured here

| measurement | result |
|---|---|
| no cache tok/s | 126 |
| kv-cache tok/s | 181 |
| C engine matches PyTorch (greedy, 150 tokens) | True |
| C engine tok/s (single CPU thread) | 729 |
| fp32 size MB / val loss | 49.3 / 1.8707 |
| int8 size MB / val loss | 17.5 / 1.8707 |

## Limitations

- One training run per stage. No variation across training seeds was measured, so small differences between checkpoints may be noise.
- The preference data for the reward model, PPO and DPO is synthetic: the chosen story mentions the requested topic and the rejected one does not. It teaches the mechanics of preference tuning; it is not human judgment of story quality.
- The topic-word metric is a word match. Story quality is not scored automatically; read the samples.
- Illustrations are class-conditioned 32×32 images from CIFAR-10, not pictures of the specific story.

## Reproduce

```bash
git checkout fa302ea
pip install -r requirements.txt   # the versions under Setup reproduce this run most closely
# then the run order in README.md, and finally:
python report.py
```
Training on different hardware or library versions will not give identical weights. With these exact checkpoints (check the SHA-256 in `manifest.json`), greedy decoding is deterministic on a given device.
