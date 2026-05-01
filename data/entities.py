# Central list of all entities to ingest.
# Add more entries freely — the system handles them automatically.

PEOPLE = [
    # Required by spec
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    # Additional people
    "Mustafa Kemal Atatürk",
    "Isaac Newton",
    "Galileo Galilei",
    "Cleopatra",
    "Genghis Khan",
    "Joan of Arc",
    "Mahatma Gandhi",
    "Martin Luther King Jr.",
    "Winston Churchill",
    "Elon Musk",
    "Barack Obama",
    "Beyoncé",
    "Stephen Hawking",
    "Bruce Lee",
    "Charlie Chaplin",
]

PLACES = [
    # Required by spec
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
    # Additional places
    "Sydney Opera House",
    "Niagara Falls",
    "Santorini",
    "Yellowstone National Park",
    "Angkor Wat",
    "Acropolis of Athens",
    "Burj Khalifa",
    "Sagrada Familia",
    "Stonehenge",
    "Louvre Museum",
    "Göbekli Tepe",
    "Banff National Park",
    "Petra",
    "Golden Gate Bridge",
    "Serengeti National Park",
    "Taşkışla",





]

# Entity type lookup (used by classifier)
PERSON_NAMES = set(PEOPLE)
PLACE_NAMES = set(PLACES)
