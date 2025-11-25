def build_podcast_dialogue(host_name: str, chef_name: str, cuisine: str, ings: List[str],
                           nutr: Dict[str,float], tags: List[str]) -> List[Tuple[str,str]]:
    ttags = ", ".join(tags) if tags else "balanced"
    hcal  = nutr["calorie_band"]; pidx = f"{nutr['protein_index']:.2f}"; hidx = f"{nutr['healthiness']:.2f}"

    return [
        ("HOST", f"Hello everyone, I'm {host_name}, and welcome back to Flavor Talks!"),
        ("HOST", f"Today we have Chef {chef_name} with us — one of the most creative voices in modern {cuisine.title()} cuisine."),
        ("CHEF", f"Hi {host_name}, thanks for having me. I'm excited to dig into today's ingredients!"),

        ("HOST", f"So Chef, before we cook, tell us a bit about the history of {cuisine.title()} cuisine. What makes it special?"),
        ("CHEF", f"{cuisine.title()} cuisine is all about balance — tradition, regional herbs, and cultural memories. Even with simple ingredients like {', '.join(ings)}, you can feel the heritage behind it."),

        ("HOST", "Beautiful. And what would you say is the heart of this dish we're about to build?"),
        ("CHEF", "Aromatics, heat control, and respecting the natural flavor of each ingredient. That’s where the magic comes from."),

        ("HOST", "Give us a quick nutritional snapshot before we jump into cooking."),
        ("CHEF", f"Sure. Calorie density is {hcal}. Protein index {pidx}. Healthiness score {hidx}. Dietary notes: {ttags}."),

        ("HOST", "Amazing. Any chef-to-chef advice for home cooks listening right now?"),
        ("CHEF", "Taste constantly. Adjust with acid, salt, and herbs — small tweaks elevate the entire dish."),

        ("HOST", "Alright Chef, walk us through how you'd turn these ingredients into something delicious."),
        ("CHEF", "Of course — let's get cooking! First, warm the pan and bloom the aromatics…"),
    ]
