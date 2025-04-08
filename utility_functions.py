def color_to_rgb(color: str) -> tuple[int, int, int]:
    if color == "blue":
        return (255, 0, 0)

    if color == "green":
        return (0, 255, 0)

    if color == "red":
        return (0, 0, 255)

    return (0, 0, 0)