# PyTorch Seq2Seq Intent Parsing

Reframing intent parsing as a human - machine translation task. Work in progress successor to [torch-seq2seq-intent-parsing](https://github.com/spro/torch-seq2seq-intent-parsing)

## The command language

This is a simple command language developed for the "home assistant" [Maia](https://github.com/withmaia) living in my apartment. She's designed as a collection of microservices with services for lights (Hue), switches (WeMo), and info such as weather and market prices.

A command consists of a "service", a "method", and some number of arguments.

```
lights setState office_light on
switches getState teapot
weather getWeather "San Francisco"
price getPrice TSLA
```

These can be represented with variable placeholders:

```
lights setState $device $state
switches getState $device
weather getWeather $location
price getPrice $symbol
```

We can imagine a bunch of human sentences that would map to a single command:

```
"Turn the office light on."
"Please turn on the light in the office."
"Maia could you set the office light on, thank you."
```

Which could similarly be represented with placeholders.

## TODO: Specific vs. freeform variables

A shortcoming of the approach so far is that the model has to learn translations of specific values, for example mapping all of the device names to their equivalent `device_name`. If we added a "basement light" the model would have no `basement_light` in the output vocabulary unless it was re-trained.

The bigger the potential input space, the more obvious the problem - consider the `getWeather` command, where the model would need to be trained with every possible location we might ask about. Worse yet, consider a `playMusic` command that could take any song or artist name...


This can be solved with a technique which I have [implemented in Torch here](https://github.com/spro/torch-seq2seq-intent-parsing). The training pairs have "variable placeholders" in the output translation, which the model generates during an intial pass. Then the network fills in the values of these placeholders with an additional pass over the input.

![](https://camo.githubusercontent.com/4125995f183d3158103b46eeb5ffdea4eef0ef52/68747470733a2f2f692e696d6775722e636f6d2f56316c747668492e706e67)
