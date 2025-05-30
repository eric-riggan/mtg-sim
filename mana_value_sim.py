import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy.random import PCG64DXSM, Generator

# constants
DECK_SIZE = 40
LAND_COUNT = 17
RNG = Generator(PCG64DXSM(42))


def get_target_card(
    mana_value: int,
    on_color_pips: int,
    off_color_pips: int,
    generic: int = 0,
) -> dict:
    """
    Get a standardized card object for the card we're evaluating.

    Parameters:
    - mana_value (int): The total mana value of the card.
    - on_color_pips (int): The number of pips that match the card's color.
    - off_color_pips (int): The number of pips that do not match the card's color.
    - generic (int, optional): The number of generic mana pips. Defaults to 0.

    Raises:
    - TypeError: If any of the parameters are not integers.
    - ValueError: If any of the parameters are out of valid range or do not sum to mana_value.

    Returns:
    - dict: A dictionary containing the calculated mana value and pips.
    """

    # Validate inputs for type
    if not isinstance(mana_value, int):
        raise TypeError("get_target_card(): mana_value must be an integer.")
    if not isinstance(on_color_pips, int):
        raise TypeError("get_target_card(): on_color_pips must be an integer.")
    if not isinstance(off_color_pips, int):
        raise TypeError("get_target_card(): off_color_pips must be an integer.")
    if not isinstance(generic, int):
        raise TypeError("get_target_card(): generic must be an integer.")
    
    # Validate inputs for value
    if mana_value < 1:
        raise ValueError("get_target_card(): mana_value must be at least 1.")
    if on_color_pips < 1:
        raise ValueError("get_target_card(): on_color_pips must be a positive integer.")
    if off_color_pips < 0:
        raise ValueError("get_target_card(): off_color_pips must be a non-negative integer.")
    if generic < 0:
        raise ValueError("get_target_card(): generic must be a non-negative integer.")
    if on_color_pips + off_color_pips + generic != mana_value:
        raise ValueError(
            "get_target_card(): The sum of on_color_pips, off_color_pips, and generic must equal mana_value."
        )

    return {
        "id": str(uuid4()),  # Unique identifier for the card
        "card_type": "target_card",
        "mana_value": mana_value,
        "on_color_pips": on_color_pips,
        "off_color_pips": off_color_pips,
        "generic": generic,
    }


def get_land(is_on_color: bool = True) -> dict:
    """
    Get a standardized land object.

    Parameters:
    - is_on_color (bool, optional): Whether the land is on color. Defaults to True.

    Raises:
    - TypeError: If is_on_color is not a boolean.

    Returns:
    - dict: A dictionary representing the land.
    """
    # Validate input
    if not isinstance(is_on_color, bool):
        raise TypeError("get_land(): is_on_color must be a boolean.")

    return {
        "id": str(uuid4()),  # Unique identifier for the land
        "card_type": "land",
        "is_on_color": is_on_color,
    }


def get_other_card() -> dict:
    """
    Get a standardized other card object.

    Returns:
    - dict: A dictionary representing a generic card.
    """
    return {
        "id": str(uuid4()),  # Unique identifier for the card
        "card_type": "other_card",
    }


def get_land_split(on_color_count: int) -> dict:
    """
    Get the split of on-color and off-color lands based on the number of on-color lands.

    Parameters:
    - on_color_count (int): The number of on-color lands.

    Raises:
    - TypeError: If on_color_count is not an integer.
    - ValueError: If on_color_count is not a positive integer.

    Returns:
    - dict: A dictionary containing the count of on-color and off-color lands.
    """
    # Validate inputs for type
    if not isinstance(on_color_count, int):
        raise TypeError("get_land_split(): on_color_count must be a positive integer.")
    
    # Validate inputs for value
    if on_color_count < 1:
        raise ValueError("get_land_split(): on_color_count must be a positive integer.")

    return {
        "on_color": on_color_count,
        "off_color": LAND_COUNT - on_color_count,
    }


def get_deck(target_card: dict, on_color_land_count: int = LAND_COUNT) -> list[dict]:
    """
    Get a deck of cards based on the target card and the number of on-color lands.
    
    Parameters:
    - target_card (dict): The target card to include in the deck.
    - on_color_land_count (int, optional): The number of on-color lands in the deck. Defaults to LAND_COUNT.

    Raises:
    - TypeError: If target_card is not a dictionary or on_color_land_count is not an integer.
    - ValueError: If the deck configuration is invalid, such as not having enough lands.
    - ValueError: If the deck size does not match the expected size.
    - ValueError: If the on_color_land_count is less than the on_color_pips of the target card.
    - ValueError: If the off_color_pips of the target card exceeds the available off-color lands.
    
    Returns:
    - list[dict]: A list of dictionaries representing the deck of cards.
    """

    # Validate arguments for type
    if not isinstance(target_card, dict):
        raise TypeError("get_deck(): target_card must be a dictionary.")
    
    if not isinstance(on_color_land_count, int):
        raise TypeError("get_deck(): on_color_land_count must be an integer.")
    

    on_color_pips = target_card["on_color_pips"]
    off_color_pips = target_card["off_color_pips"]

    if on_color_land_count < on_color_pips:
        raise ValueError(
            f"Invalid deck configuration: Not enough on-color lands: {on_color_land_count} < {on_color_pips}"
        )
    if LAND_COUNT - on_color_land_count < off_color_pips:
        raise ValueError(
            f"Invalid deck configuration: Not enough off-color lands: {LAND_COUNT - on_color_land_count} < {off_color_pips}"
        )

    split = get_land_split(on_color_land_count)

    deck = []

    # Add on-color lands
    for _ in range(split["on_color"]):
        deck.append(get_land(is_on_color=True))

    # Add off-color lands
    for _ in range(split["off_color"]):
        deck.append(get_land(is_on_color=False))

    # Add other cards
    for _ in range(DECK_SIZE - LAND_COUNT - 1):
        deck.append(get_other_card())

    # Do not add the target card - we'll add it to the player's hand directly.

    # Check deck size, should be DECK_SIZE - 1
    if len(deck) != DECK_SIZE - 1:
        raise ValueError(f"Deck size mismatch: {len(deck)} != {DECK_SIZE - 1}")

    # Shuffle the deck
    RNG.shuffle(deck)
    return deck


def draw_n(
    deck: list[dict],
    n: int,
) -> tuple[list[dict], list[dict]]:
    """
    Draw n cards from the deck.

    Parameters:
    - deck (list[dict]): The deck of cards to draw from.
    - n (int): The number of cards to draw.

    Raises:
    - ValueError: If n is less than 1, or if the deck is not a list of dictionaries,
    - ValueError: If n is greater than the size of the deck.
    - TypeError: If deck is not a list or if any card in the deck is not a dictionary.
    
    Returns:
    - tuple: A tuple containing two lists:
        - The first list contains the drawn cards.
        - The second list contains the remaining cards in the deck.
    """

    # validate arguments
    if n < 1:
        raise ValueError("Cannot draw less than 1 card.")

    if not isinstance(deck, list):
        raise TypeError("Deck must be a list of cards.")
    if not all(isinstance(card, dict) for card in deck):
        raise TypeError("All cards in the deck must be dictionaries.")

    if n > len(deck):
        raise ValueError(f"Cannot draw {n} cards from a deck of size {len(deck)}.")

    cards_drawn = deck[:n]
    deck = deck[n:]
    return cards_drawn, deck


def simulate(
    mana_value: int,
    on_color_pips: int,
    off_color_pips: int,
    on_color_land_count: int = LAND_COUNT,
):
    """
    Simulate a game of Magic: The Gathering to evaluate the average turn
    on which a card with the specified mana value and pips can be cast, given
    a deck containing n on-color lands.

    Parameters:
    - mana_value (int): The total mana value of the card.
    - on_color_pips (int): The number of pips that match the card's color.
    - off_color_pips (int): The number of pips that do not match the card's color.
    - on_color_land_count (int, optional): The number of on-color lands in the deck. Defaults to LAND_COUNT.

    Raises:
    - TypeError: If any of the parameters are not integers.
    - ValueError: If any of the parameters are out of valid range or do not sum to mana_value.
    - ValueError: If the on_color_land_count is less than 1 or greater than LAND_COUNT.
    - ValueError: If the mana value is too low for the given pips.
    - ValueError: If the mana value does not match the sum of generic, on_color_pips, and off_color_pips.
    - ValueError: If the hand size or deck size does not match the expected sizes after mulligans.

    Returns:
    - dict: A dictionary containing the results of the simulation, including:
        - cast_turn (float): The turn on which the card was cast, or None if it was never cast.
        - mana_value (int): The mana value of the target card.
        - on_color_pips (int): The number of on-color pips of the target card.
        - off_color_pips (int): The number of off-color pips of the target card.
        - generic (int): The number of generic pips of the target card.
        - mulligan_count (float): The number of mulligans taken.
    """

    # Validate inputs for type
    if not isinstance(mana_value, int):
        raise TypeError("Mana value must be an integer.")
    if not isinstance(on_color_pips, int):
        raise TypeError("On-color pips must be an integer.")
    if not isinstance(off_color_pips, int):
        raise TypeError("Off-color pips must be an integer.")
    if not isinstance(on_color_land_count, int):
        raise TypeError("On-color land count must be an integer.")
    
    # Validate inputs for value
    if mana_value < 1:
        raise ValueError("Mana value must be at least 1.")
    if on_color_pips < 0:
        raise ValueError("On-color pips must be a non-negative integer.")
    if off_color_pips < 0:
        raise ValueError("Off-color pips must be a non-negative integer.")
    if on_color_land_count < 1:
        raise ValueError("On-color land count must be at least 1.")
    if on_color_land_count > LAND_COUNT:
        raise ValueError(
            f"On-color land count must be at most {LAND_COUNT}, got {on_color_land_count}."
        )

    generic = mana_value - on_color_pips - off_color_pips
    if generic < 0:
        raise ValueError("Mana value is too low for the given pips.")

    if generic + on_color_pips + off_color_pips != mana_value:
        raise ValueError(
            f"Mana value mismatch: {generic} + {on_color_pips} + {off_color_pips} != {mana_value}"
        )

    target_card = get_target_card(
        mana_value=mana_value,
        on_color_pips=on_color_pips,
        off_color_pips=off_color_pips,
        generic=generic,
    )

    deck = get_deck(target_card=target_card, on_color_land_count=on_color_land_count)

    # Draw an opening hand of 6 cards
    hand, deck = draw_n(deck, 6)

    # Add the target card to the hand
    hand.append(target_card)

    # Mulligan heuristics:
    # 1. Mull to 6 or 5 if we have no on-color lands
    # 2. Mull to 4 if we have no lands at all
    # 3. Do not mulligan lower than 4

    mulligan_count = 0
    while True:
        on_color_land_count = sum(
            1 for card in hand if card["card_type"] == "land" and card["is_on_color"]
        )
        off_color_land_count = sum(
            1
            for card in hand
            if card["card_type"] == "land" and not card["is_on_color"]
        )

        # 7 or 6 card hand
        if mulligan_count in (0, 1) and on_color_land_count == 0:
            # remove the target card from hand
            hand.remove(target_card)
            # put the rest of the cards back into the deck
            deck.extend(hand)
            # draw a new hand of 6 cards
            hand, deck = draw_n(deck, 6)
            # add the target card back to the hand
            hand.append(target_card)
        # 5 card hand
        elif mulligan_count == 2 and (on_color_land_count + off_color_land_count) == 0:
            # remove the target card from hand
            hand.remove(target_card)
            # put the rest of the cards back into the deck
            deck.extend(hand)
            # draw a new hand of 6 cards
            hand, deck = draw_n(deck, 6)
            # add the target card back to the hand
            hand.append(target_card)
        # 4 card hand
        elif mulligan_count == 3:
            # Do not mulligan lower than 4
            break
        else:
            # No mulligan needed, break the loop
            break

        mulligan_count += 1

    # return cards to the deck equal to the mulligan count. Prioritize
    # other cards, followed by off-color lands, then on-color lands.
    to_return = []

    while True:
        if mulligan_count == 0:
            break

        if len(to_return) == mulligan_count:
            break

        if any(card["card_type"] == "other_card" for card in hand):
            return_card = next(
                card for card in hand if card["card_type"] == "other_card"
            )
            to_return.append(return_card)
            hand.remove(return_card)
            continue

        if any(
            card["card_type"] == "land" and not card["is_on_color"] for card in hand
        ):
            return_card = next(
                card
                for card in hand
                if card["card_type"] == "land" and not card["is_on_color"]
            )
            to_return.append(return_card)
            hand.remove(return_card)
            continue

        if any(card["card_type"] == "land" and card["is_on_color"] for card in hand):
            return_card = next(
                card
                for card in hand
                if card["card_type"] == "land" and card["is_on_color"]
            )
            to_return.append(return_card)
            hand.remove(return_card)
            continue

    # Add the returned cards to the deck
    deck.extend(to_return)

    # We should now have a hand of 7 - mulligan_count cards, and a deck
    # of DECK_SIZE - 1 - mulligan_count cards.

    # Check hand size
    if len(hand) != 7 - mulligan_count:
        raise ValueError(f"Hand size mismatch: {len(hand)} != {7 - mulligan_count}")

    # Check deck size
    if len(deck) != DECK_SIZE - len(hand):
        raise ValueError(
            f"Deck size mismatch: {len(deck)} != {DECK_SIZE - len(hand)} ({DECK_SIZE} - {len(hand)})"
        )

    # Begin playing the game.
    # RULES
    # 1. We can play one land per turn. Prioritize on-color lands, or off-color if we already have an on-color land.
    # 2. We can cast the target card if we have enough mana.
    #    - We must have at least as many on-color pips as the target card's on-color pips.
    #    - We must have at least as many off-color pips as the target card's off-color pips.
    # 3. Once we can cast the target card, we do so immediately and end the game.

    turn = 1

    board = []
    while True:
        land_drop = False

        # Draw a card
        if len(deck) > 0 and turn > 1:
            drawn_card, deck = draw_n(deck, 1)
            hand.append(drawn_card[0])
        elif len(deck) == 0:
            # No more cards to draw, we can't continue
            break

        # Check if we have any lands in hand
        have_lands = any(card["card_type"] == "land" for card in hand)
        hand_has_on_color_land = any(
            card["card_type"] == "land" and card["is_on_color"] for card in hand
        )
        hand_has_off_color_land = any(
            card["card_type"] == "land" and not card["is_on_color"] for card in hand
        )

        board_on_color_count = sum(
            1 for card in board if card.get("is_on_color", False)
        )
        board_off_color_count = sum(
            1 for card in board if not card.get("is_on_color", False)
        )
        on_color_req = target_card["on_color_pips"]
        off_color_req = target_card["off_color_pips"]

        if have_lands is True:
            # Prioritize on-color lands until we have enough on-color pips
            # Then prioritize off-color lands until we have enough off-color pips
            # Then play any land we can.

            # Prioritize on-color lands
            if (
                hand_has_on_color_land
                and board_on_color_count < on_color_req
                and land_drop is False
            ):
                land = next(
                    card
                    for card in hand
                    if card["card_type"] == "land" and card["is_on_color"]
                )
                board.append(land)
                hand.remove(land)
                land_drop = True
            # Prioritize off-color lands
            elif (
                hand_has_off_color_land
                and board_off_color_count < off_color_req
                and land_drop is False
            ):
                land = next(
                    card
                    for card in hand
                    if card["card_type"] == "land" and not card["is_on_color"]
                )
                board.append(land)
                hand.remove(land)
                land_drop = True
            # Play any land we can
            elif (
                any(card["card_type"] == "land" for card in hand) and land_drop is False
            ):
                land = next(card for card in hand if card["card_type"] == "land")
                board.append(land)
                hand.remove(land)
                land_drop = True
        else:
            # No lands in hand, we can't play a land this turn
            land_drop = False

        # Recount on-color and off-color lands on the board
        board_on_color_count = sum(
            1 for card in board if card.get("is_on_color", False)
        )
        board_off_color_count = sum(
            1 for card in board if not card.get("is_on_color", False)
        )

        # Check if we can cast the target card
        if (
            # Do we have enough on-color and off-color pips?
            board_on_color_count >= on_color_req
            and board_off_color_count >= off_color_req
            and
            # Do we have enough total total lands to cast the target card?
            (board_on_color_count + board_off_color_count) >= target_card["mana_value"]
        ):
            # We can cast the target card
            return {
                "cast_turn": np.float64(turn),
                "mana_value": np.int16(target_card["mana_value"]),
                "on_color_pips": np.int16(target_card["on_color_pips"]),
                "off_color_pips": np.int16(target_card["off_color_pips"]),
                "generic": np.int16(target_card["generic"]),
                "mulligan_count": np.float64(mulligan_count),
            }
        else:
            # We cannot cast the target card yet, continue to the next turn
            turn += 1
    # If we exit the loop, we never cast the target card
    return {
        "cast_turn": None,
        "mana_value": np.int16(target_card["mana_value"]),
        "on_color_pips": np.int16(target_card["on_color_pips"]),
        "off_color_pips": np.int16(target_card["off_color_pips"]),
        "generic": np.int16(target_card["generic"]),
        "mulligan_count": np.float64(mulligan_count),
    }

def run_simulations(
    mana_value: int,
    on_color_pips: int,
    off_color_pips: int,
    on_color_land_count: int = LAND_COUNT,
    num_simulations: int = 10000,
    verbose: bool = False,
) -> dict:
    """
    Run multiple simulations of the game using multithreading and return the
    results as a summary dict.

    Parameters:
    - mana_value (int): The total mana value of the card.
    - on_color_pips (int): The number of pips that match the card's color.
    - off_color_pips (int): The number of pips that do not match the card's color.
    - on_color_land_count (int, optional): The number of on-color lands in the deck. Defaults to LAND_COUNT.
    - num_simulations (int, optional): The number of simulations to run. Defaults to 10000.

    Raises:
    - TypeError: If any of the parameters are not integers.
    - ValueError: If any of the parameters are out of valid range or do not sum to mana_value.
    - ValueError: If the on_color_land_count is less than 1 or greater than LAND_COUNT.
    - ValueError: If the mana value is too low for the given pips.
    - ValueError: If the mana value does not match the sum of generic, on_color_pips, and off_color_pips.
    - ValueError: If the number of simulations is less than 1.

    Returns:
    - dict: A dictionary containing the results of the simulations, including:
        - mana_value (int): The mana value of the target card.
        - on_color_pips (int): The number of on-color pips of the target card.
        - off_color_pips (int): The number of off-color pips of the target card.
        - on_color_land_count (int): The number of on-color lands in the deck.
        - off_color_land_count (int): The number of off-color lands in the deck.
        - iterations (int): The number of simulations run.
        - average_cast_turn (float): The average turn on which the card was cast.
        - average_mulligan_count (float): The average number of mulligans taken.
    """
    # Validate inputs for type
    if not isinstance(mana_value, int):
        raise TypeError("Mana value must be an integer.")
    if not isinstance(on_color_pips, int):
        raise TypeError("On-color pips must be an integer.")
    if not isinstance(off_color_pips, int):
        raise TypeError("Off-color pips must be an integer.")
    if not isinstance(on_color_land_count, int):
        raise TypeError("On-color land count must be an integer.")
    if not isinstance(num_simulations, int):
        raise TypeError("Number of simulations must be an integer.")
    
    # Validate inputs for value
    if mana_value < 1:
        raise ValueError("Mana value must be at least 1.")
    if on_color_pips < 1:
        raise ValueError("On-color pips must be a positive integer.")
    if off_color_pips < 0:
        raise ValueError("Off-color pips must be a non-negative integer.")
    if on_color_land_count < 1:
        raise ValueError("On-color land count must be at least 1.")
    if on_color_land_count > LAND_COUNT:
        raise ValueError(
            f"On-color land count must be at most {LAND_COUNT}, got {on_color_land_count}."
        )
    if num_simulations < 1:
        raise ValueError("Number of simulations must be at least 1.")
    if on_color_pips + off_color_pips > mana_value:
        raise ValueError(
            "The sum of on_color_pips and off_color_pips must not exceed mana_value."
        )

    # Determine thread pool size based on CPU count
    max_workers = min(16, (os.cpu_count() or 1) + 4)  # Cap at 16 for safety

    if verbose is True:
        print(f"Mana Value: {mana_value}, On-Color Pips: {on_color_pips}, "
              f"Off-Color Pips: {off_color_pips}, On-Color Land Count: {on_color_land_count}. \n\n"
              f"Running {num_simulations} simulations with {max_workers} threads...", end="\r")

    def worker(_):
        try:
            return simulate(
                mana_value=mana_value,
                on_color_pips=on_color_pips,
                off_color_pips=off_color_pips,
                on_color_land_count=on_color_land_count,
            )
        except Exception as e:
            # If an error occurs, return None
            return None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(num_simulations)]
        for future in as_completed(futures):
            fut_result = future.result()
            if fut_result is not None:
                results.append(fut_result)

    df = pd.DataFrame(results)
    if len(df) > 0:
        return {
            "mana_value": mana_value,
            "on_color_pips": on_color_pips,
            "off_color_pips": off_color_pips,
            "on_color_land_count": on_color_land_count,
            "off_color_land_count": LAND_COUNT - on_color_land_count,
            "iterations": num_simulations,
            "average_cast_turn": df["cast_turn"].mean(),
            "average_mulligan_count": df["mulligan_count"].mean(),
        }
    else:
        return None


def main(
    iterations_per_simulation: int = 10000,
    min_mana_value: int = 1,
    max_mana_value: int = 7,
    include_gold_cards: bool = True,  # If False, off_color_pips will always be 0
    verbose: bool = False,
):
    """
    Main function to run the mana value simulations and save the results to a CSV file.

    Parameters:
    - iterations_per_simulation (int, optional): The number of simulations to run for each mana value. Defaults to 10000.
    - min_mana_value (int, optional): The minimum mana value to simulate. Defaults to 1.
    - max_mana_value (int, optional): The maximum mana value to simulate. Defaults to 7.
    - include_gold_cards (bool, optional): Whether to include gold cards in the simulation. Defaults to True.

    Returns:
    - None: The function saves the results to a CSV file and does not return anything.
    """
    collated_data = []

    for on_color_land_count in range(1, LAND_COUNT + 1):
        for mana_value in range(min_mana_value, max_mana_value + 1):
            # If we're not including gold cards, set off_color_pips to 0
            if include_gold_cards is False:
                for on_color_pips in range(1, mana_value + 1):
                    off_color_pips = 0
                    result = run_simulations(
                        mana_value=mana_value,
                        on_color_pips=on_color_pips,
                        off_color_pips=off_color_pips,
                        on_color_land_count=on_color_land_count,
                        num_simulations=iterations_per_simulation,
                    )
                    if result is not None:
                        collated_data.append(result)
            else:
                # If we're including gold cards, get ever combination of on and off
                # color pips within the mana value. For example, if mana_value is 3,
                # (assume on_color == W and off_color == U), we can have:
                # 2W, 2U, 1WU, 1WW, 1UU, WWU, UUW, WWW, and UUU. The combinations of
                # (on_color_pips, off_color_pips) are:
                # (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (3, 0), (0, 3)

                for on_color_pips in range(1, mana_value + 1):
                    for off_color_pips in range(0, mana_value - on_color_pips + 1):
                        if on_color_pips == 0 and off_color_pips == 0:
                            continue
                        elif on_color_pips + off_color_pips > mana_value:
                            continue
                        else:
                            try:
                                result = run_simulations(
                                    mana_value=mana_value,
                                    on_color_pips=on_color_pips,
                                    off_color_pips=off_color_pips,
                                    on_color_land_count=on_color_land_count,
                                    num_simulations=iterations_per_simulation,
                                    verbose=verbose,
                                )
                            except Exception as e:
                                result = None
                            if result is not None:
                                collated_data.append(result)

    # Convert the collated data to a DataFrame
    df = pd.DataFrame(collated_data)
    # Save the DataFrame to a CSV file
    output_file = "mana_value_simulation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Simulation results saved to {output_file}")


if __name__ == "__main__":
    main(
        iterations_per_simulation=100000,
        min_mana_value=1,
        max_mana_value=7,
        include_gold_cards=True,
        verbose=True,
    )
