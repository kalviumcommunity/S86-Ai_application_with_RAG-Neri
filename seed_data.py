from pathlib import Path
from datetime import datetime, timedelta
import random

from faker import Faker

from src.config import (
    MANUALS_DIR,
    MAINTENANCE_LOGS_DIR,
    SAFETY_DIR,
)
from src.embeddings import create_embeddings
from src.ingestion import create_chunks
from src.vector_store import add_documents
from src.history import (
    initialize_database,
    save_document,
    get_documents,
)


# ============================================================
# Configuration
# ============================================================

fake = Faker()

random.seed(42)
Faker.seed(42)

MAINTENANCE_RECORDS_PER_MACHINE = 25


# ============================================================
# Machine Definitions
# ============================================================

MACHINES = [
    {
        "machine": "CNC Mill",
        "machine_id": "CNC-001",
        "manufacturer": "Haas",
        "model": "VF-2",
        "department": "Machining",
        "location": "Plant A - Bay 1",
        "manual_owner": "Maintenance Department",
        "safety_owner": "Safety Department",
        "problems": [
            "overheating",
            "excessive vibration",
            "coolant flow issue",
            "spindle not starting",
            "unexpected shutdown",
            "tool wear",
            "poor surface finish",
            "abnormal noise",
        ],
    },
    {
        "machine": "CNC Lathe",
        "machine_id": "LAT-001",
        "manufacturer": "DMG Mori",
        "model": "NLX 2500",
        "department": "Machining",
        "location": "Plant A - Bay 2",
        "manual_owner": "Maintenance Department",
        "safety_owner": "Safety Department",
        "problems": [
            "excessive vibration",
            "chuck problem",
            "spindle overheating",
            "poor surface finish",
            "tool alignment issue",
            "unexpected shutdown",
        ],
    },
    {
        "machine": "Hydraulic Press",
        "machine_id": "PRS-001",
        "manufacturer": "Schuler",
        "model": "HP-160",
        "department": "Forming",
        "location": "Plant A - Bay 4",
        "manual_owner": "Maintenance Department",
        "safety_owner": "Safety Department",
        "problems": [
            "low pressure",
            "pressure fluctuation",
            "hydraulic leak",
            "slow operation",
            "unexpected shutdown",
            "overheating",
        ],
    },
    {
        "machine": "Industrial Drill",
        "machine_id": "DRL-001",
        "manufacturer": "Bosch Rexroth",
        "model": "ID-500",
        "department": "Fabrication",
        "location": "Plant B - Bay 1",
        "manual_owner": "Maintenance Department",
        "safety_owner": "Safety Department",
        "problems": [
            "drill bit slipping",
            "excessive vibration",
            "motor overheating",
            "poor drilling accuracy",
            "abnormal noise",
            "unexpected shutdown",
        ],
    },
    {
        "machine": "Milling Machine",
        "machine_id": "MIL-001",
        "manufacturer": "Bridgeport",
        "model": "Series 1",
        "department": "Machining",
        "location": "Plant B - Bay 3",
        "manual_owner": "Maintenance Department",
        "safety_owner": "Safety Department",
        "problems": [
            "spindle vibration",
            "poor surface finish",
            "tool wear",
            "spindle overheating",
            "coolant issue",
            "unexpected shutdown",
        ],
    },
    {
        "machine": "Surface Grinder",
        "machine_id": "GRD-001",
        "manufacturer": "Okamoto",
        "model": "ACC-63",
        "department": "Finishing",
        "location": "Plant B - Bay 5",
        "manual_owner": "Maintenance Department",
        "safety_owner": "Safety Department",
        "problems": [
            "grinding wheel vibration",
            "poor surface finish",
            "wheel wear",
            "coolant flow issue",
            "motor overheating",
            "unexpected shutdown",
        ],
    },
    {
        "machine": "Conveyor System",
        "machine_id": "CON-001",
        "manufacturer": "Dorner",
        "model": "2200 Series",
        "department": "Material Handling",
        "location": "Plant C - Bay 1",
        "manual_owner": "Maintenance Department",
        "safety_owner": "Safety Department",
        "problems": [
            "belt slipping",
            "belt misalignment",
            "motor overheating",
            "abnormal noise",
            "conveyor not starting",
            "unexpected shutdown",
        ],
    },
    {
        "machine": "Injection Molding Machine",
        "machine_id": "IMM-001",
        "manufacturer": "Engel",
        "model": "Victory 330",
        "department": "Molding",
        "location": "Plant C - Bay 3",
        "manual_owner": "Maintenance Department",
        "safety_owner": "Safety Department",
        "problems": [
            "injection pressure low",
            "material temperature high",
            "hydraulic pressure low",
            "poor molding quality",
            "machine not starting",
            "unexpected shutdown",
        ],
    },
    {
        "machine": "Robotic Welding Cell",
        "machine_id": "ROB-001",
        "manufacturer": "ABB",
        "model": "IRB 2600",
        "department": "Welding",
        "location": "Plant C - Bay 5",
        "manual_owner": "Maintenance Department",
        "safety_owner": "Safety Department",
        "problems": [
            "welding arc unstable",
            "robot movement error",
            "wire feeding issue",
            "excessive vibration",
            "controller alarm",
            "unexpected shutdown",
        ],
    },
    {
        "machine": "Air Compressor",
        "machine_id": "CMP-001",
        "manufacturer": "Atlas Copco",
        "model": "GA 30",
        "department": "Utilities",
        "location": "Plant D - Utility Room",
        "manual_owner": "Maintenance Department",
        "safety_owner": "Safety Department",
        "problems": [
            "low air pressure",
            "high discharge temperature",
            "oil leak",
            "abnormal noise",
            "compressor not starting",
            "unexpected shutdown",
        ],
    },
]


# ============================================================
# Controlled Technical Information
# ============================================================

TECHNICAL_DATA = {
    "CNC Mill": {
        "manual": {
            "overheating": [
                "Stop the machine and allow it to cool before inspection.",
                "Check the coolant level and confirm coolant is circulating.",
                "Inspect the cooling system for blockage or restricted airflow.",
                "Check the maintenance log for previous overheating events.",
            ],
            "excessive vibration": [
                "Stop machining before inspecting the machine.",
                "Inspect the tool holder and cutting tool for visible damage.",
                "Check whether the maintenance log records previous vibration issues.",
                "Verify that the workholding setup is secure according to the machine procedure.",
            ],
            "coolant flow issue": [
                "Stop the machine before inspecting the coolant system.",
                "Check the coolant reservoir level.",
                "Inspect accessible coolant lines for blockage.",
                "Review previous maintenance records for coolant-system issues.",
            ],
            "spindle not starting": [
                "Confirm the machine is in a safe stopped state.",
                "Check the operator panel for an active alarm.",
                "Review the maintenance record for previous spindle faults.",
                "Do not inspect internal electrical components while connected to power.",
            ],
            "unexpected shutdown": [
                "Record the displayed alarm or error code.",
                "Do not immediately restart the machine repeatedly.",
                "Review recent maintenance records for shutdown events.",
                "Follow the approved restart procedure in the machine manual.",
            ],
            "tool wear": [
                "Stop the machining operation before inspecting the tool.",
                "Inspect the cutting tool for visible wear or damage.",
                "Review recent maintenance records for tool-related issues.",
                "Replace tooling only according to the approved machine procedure.",
            ],
            "poor surface finish": [
                "Stop the operation if the surface quality continues to deteriorate.",
                "Inspect the cutting tool for wear.",
                "Check the maintenance history for previous surface-finish problems.",
                "Verify that the workholding setup is secure.",
            ],
            "abnormal noise": [
                "Stop the machine if the abnormal noise is unexpected or increasing.",
                "Record when the noise occurs during operation.",
                "Inspect accessible tooling and workholding components.",
                "Review maintenance records for similar noise reports.",
            ],
        },
        "safety": [
            "Stop the CNC mill before performing maintenance or inspection.",
            "Do not inspect internal electrical components while the machine is connected to power.",
            "Follow the organization's lockout and tagout procedure before accessing hazardous internal components.",
            "Keep the machine area clear of unauthorized personnel during maintenance.",
            "If smoke, fire, sparking, or electrical damage is observed, follow the approved emergency procedure.",
        ],
    },

    "CNC Lathe": {
        "manual": {
            "excessive vibration": [
                "Stop the machining operation before inspection.",
                "Inspect the chuck and workholding setup for visible problems.",
                "Check the cutting tool for wear or damage.",
                "Review previous maintenance records for vibration events.",
            ],
            "chuck problem": [
                "Stop the spindle before inspecting the chuck.",
                "Check the chuck condition using the approved inspection procedure.",
                "Verify that the workpiece is securely held.",
                "Review maintenance history for previous chuck issues.",
            ],
            "spindle overheating": [
                "Stop the machine and allow the spindle to cool.",
                "Check the cooling system according to the manual.",
                "Review maintenance records for previous spindle temperature problems.",
                "Do not access internal components while energized.",
            ],
            "poor surface finish": [
                "Inspect the cutting tool for wear.",
                "Check workholding stability.",
                "Review recent maintenance records.",
                "Verify the tool alignment according to the approved procedure.",
            ],
            "tool alignment issue": [
                "Stop machining before checking tool alignment.",
                "Inspect the tool holder and cutting tool.",
                "Verify alignment using the approved machine procedure.",
                "Review previous alignment-related maintenance records.",
            ],
            "unexpected shutdown": [
                "Record the displayed alarm or error code.",
                "Do not repeatedly restart the machine without checking the cause.",
                "Review recent maintenance records.",
                "Follow the approved restart procedure.",
            ],
        },
        "safety": [
            "Stop the spindle before inspecting the chuck or workholding system.",
            "Keep hands away from rotating components.",
            "Use the approved lockout and tagout procedure before maintenance requiring access to hazardous components.",
            "Do not bypass machine safety interlocks.",
            "If electrical sparking, smoke, or fire occurs, follow the approved emergency procedure.",
        ],
    },

    "Hydraulic Press": {
        "manual": {
            "low pressure": [
                "Stop the press before performing inspection.",
                "Check the hydraulic fluid level.",
                "Inspect accessible hydraulic connections for visible leakage.",
                "Review maintenance history for previous pressure problems.",
            ],
            "pressure fluctuation": [
                "Stop operation if pressure becomes unstable.",
                "Check hydraulic fluid level according to the manual.",
                "Review the maintenance log for previous pressure fluctuations.",
                "Inspect accessible hydraulic lines for visible problems.",
            ],
            "hydraulic leak": [
                "Stop the machine before inspecting the hydraulic system.",
                "Identify the visible location of the leak without contacting hydraulic fluid.",
                "Review maintenance records for previous hydraulic leaks.",
                "Do not open pressurized hydraulic components unless authorized.",
            ],
            "slow operation": [
                "Stop the press if movement is significantly slower than normal.",
                "Check hydraulic fluid level.",
                "Review maintenance records for previous slow-operation events.",
                "Inspect accessible hydraulic lines for visible leakage.",
            ],
            "unexpected shutdown": [
                "Record the machine alarm or error code.",
                "Keep the press in a safe stopped condition.",
                "Review recent maintenance records.",
                "Follow the approved restart procedure.",
            ],
            "overheating": [
                "Stop the press and allow the system to cool.",
                "Check hydraulic fluid level.",
                "Inspect accessible cooling components.",
                "Review maintenance history for previous overheating events.",
            ],
        },
        "safety": [
            "Never place hands or body parts inside the press working area.",
            "Use the approved lockout and tagout procedure before maintenance.",
            "Do not inspect pressurized hydraulic components while the system is energized.",
            "Keep unauthorized personnel away from the press during maintenance.",
            "If a hydraulic leak creates a hazardous condition, stop operation and follow site safety procedures.",
        ],
    },

    "Industrial Drill": {
        "manual": {
            "drill bit slipping": [
                "Stop the drill before inspecting the tool holder.",
                "Check the drill bit and chuck for visible damage.",
                "Verify that the drill bit is secured according to the operating procedure.",
                "Review maintenance records for previous chuck problems.",
            ],
            "excessive vibration": [
                "Stop drilling before inspection.",
                "Inspect the drill bit for damage or excessive wear.",
                "Check the workpiece setup.",
                "Review previous vibration-related maintenance records.",
            ],
            "motor overheating": [
                "Stop the drill and allow the motor to cool.",
                "Check for blocked ventilation around the motor.",
                "Review maintenance records for previous overheating events.",
                "Do not access internal electrical components while energized.",
            ],
            "poor drilling accuracy": [
                "Stop the operation if accuracy continues to degrade.",
                "Inspect the drill bit for wear.",
                "Check the workpiece setup.",
                "Review maintenance history for alignment problems.",
            ],
            "abnormal noise": [
                "Stop the drill if abnormal noise increases.",
                "Record when the noise occurs.",
                "Inspect accessible tooling for visible damage.",
                "Review previous maintenance records.",
            ],
            "unexpected shutdown": [
                "Record the displayed alarm.",
                "Keep the machine stopped until the cause is reviewed.",
                "Check recent maintenance records.",
                "Follow the approved restart procedure.",
            ],
        },
        "safety": [
            "Stop the drill before changing or inspecting tooling.",
            "Keep loose clothing and hands away from rotating components.",
            "Use approved eye protection in the drilling area.",
            "Use lockout and tagout before maintenance requiring access to hazardous components.",
            "If electrical smoke or sparking occurs, stop operation and follow the emergency procedure.",
        ],
    },

    "Milling Machine": {
        "manual": {
            "spindle vibration": [
                "Stop the spindle before inspection.",
                "Inspect the tool holder and cutting tool.",
                "Check the workholding setup.",
                "Review maintenance records for previous vibration events.",
            ],
            "poor surface finish": [
                "Inspect the cutting tool for wear.",
                "Check workholding stability.",
                "Review maintenance history for similar problems.",
                "Verify tooling according to the approved procedure.",
            ],
            "tool wear": [
                "Stop the operation before inspecting the tool.",
                "Inspect the cutting tool for visible wear.",
                "Review recent maintenance records.",
                "Replace tooling according to the approved procedure.",
            ],
            "spindle overheating": [
                "Stop the spindle and allow it to cool.",
                "Check the cooling system.",
                "Review maintenance records.",
                "Do not access internal components while energized.",
            ],
            "coolant issue": [
                "Stop machining before inspecting the coolant system.",
                "Check coolant level.",
                "Inspect accessible coolant lines.",
                "Review previous coolant-related maintenance records.",
            ],
            "unexpected shutdown": [
                "Record the machine alarm.",
                "Do not repeatedly restart the machine.",
                "Review maintenance history.",
                "Follow the approved restart procedure.",
            ],
        },
        "safety": [
            "Stop the spindle before inspecting tooling.",
            "Keep hands away from rotating components.",
            "Use lockout and tagout before authorized maintenance.",
            "Do not bypass machine guards or safety interlocks.",
            "Follow approved emergency procedures for smoke, fire, or electrical faults.",
        ],
    },

    "Surface Grinder": {
        "manual": {
            "grinding wheel vibration": [
                "Stop the grinder before inspection.",
                "Inspect the grinding wheel for visible damage.",
                "Check the wheel mounting condition according to the approved procedure.",
                "Review maintenance records for previous vibration issues.",
            ],
            "poor surface finish": [
                "Stop the operation if surface quality deteriorates.",
                "Inspect the grinding wheel condition.",
                "Review recent maintenance history.",
                "Verify coolant flow according to the machine procedure.",
            ],
            "wheel wear": [
                "Stop the machine before inspecting the wheel.",
                "Inspect the wheel condition.",
                "Review maintenance records.",
                "Replace the wheel only according to the approved procedure.",
            ],
            "coolant flow issue": [
                "Stop grinding before inspecting the coolant system.",
                "Check coolant level.",
                "Inspect accessible coolant lines.",
                "Review previous maintenance records.",
            ],
            "motor overheating": [
                "Stop the grinder and allow the motor to cool.",
                "Check ventilation around the motor.",
                "Review maintenance history.",
                "Do not access internal electrical components while energized.",
            ],
            "unexpected shutdown": [
                "Record the alarm or error code.",
                "Keep the machine stopped.",
                "Review recent maintenance records.",
                "Follow the approved restart procedure.",
            ],
        },
        "safety": [
            "Stop the grinder before inspecting the grinding wheel.",
            "Do not operate the machine with a damaged grinding wheel.",
            "Use approved eye and face protection.",
            "Use lockout and tagout before authorized maintenance.",
            "Follow emergency procedures if wheel damage, fire, smoke, or sparking occurs.",
        ],
    },

    "Conveyor System": {
        "manual": {
            "belt slipping": [
                "Stop the conveyor before inspection.",
                "Inspect the belt and accessible drive components.",
                "Check the maintenance log for previous belt-slip events.",
                "Adjust or repair the belt only according to the approved procedure.",
            ],
            "belt misalignment": [
                "Stop the conveyor before inspection.",
                "Inspect belt tracking and accessible rollers.",
                "Review previous maintenance records.",
                "Correct alignment according to the approved procedure.",
            ],
            "motor overheating": [
                "Stop the conveyor and allow the motor to cool.",
                "Check ventilation around the motor.",
                "Review maintenance history.",
                "Do not access internal electrical components while energized.",
            ],
            "abnormal noise": [
                "Stop the conveyor if the noise is unexpected or increasing.",
                "Inspect accessible rollers and belt components.",
                "Review maintenance records.",
                "Do not enter guarded areas without following safety procedures.",
            ],
            "conveyor not starting": [
                "Confirm the conveyor is in a safe stopped state.",
                "Check the operator panel for alarms.",
                "Review recent maintenance records.",
                "Follow the approved restart procedure.",
            ],
            "unexpected shutdown": [
                "Record the alarm or error condition.",
                "Keep the conveyor stopped until the cause is identified.",
                "Review maintenance history.",
                "Follow the approved restart procedure.",
            ],
        },
        "safety": [
            "Never reach into a moving conveyor.",
            "Stop the conveyor before inspecting belts or rollers.",
            "Use lockout and tagout before entering guarded maintenance areas.",
            "Do not bypass emergency stops or safety guards.",
            "Follow emergency procedures for electrical faults, fire, or entrapment hazards.",
        ],
    },

    "Injection Molding Machine": {
        "manual": {
            "injection pressure low": [
                "Stop the machine if pressure remains outside the approved range.",
                "Check hydraulic fluid level according to the manual.",
                "Review maintenance records for previous pressure problems.",
                "Inspect accessible hydraulic components for visible leakage.",
            ],
            "material temperature high": [
                "Stop production if material temperature exceeds the approved range.",
                "Check the temperature display and cooling system.",
                "Review maintenance records for previous temperature issues.",
                "Do not access hot internal components without authorization.",
            ],
            "hydraulic pressure low": [
                "Stop the machine before inspection.",
                "Check hydraulic fluid level.",
                "Inspect accessible hydraulic connections for visible leakage.",
                "Review maintenance history.",
            ],
            "poor molding quality": [
                "Stop production if product quality continues to deteriorate.",
                "Record the observed quality problem.",
                "Review maintenance history for previous molding issues.",
                "Check approved machine parameters.",
            ],
            "machine not starting": [
                "Check the operator panel for alarms.",
                "Confirm the machine is in the correct operating state.",
                "Review recent maintenance records.",
                "Follow the approved startup procedure.",
            ],
            "unexpected shutdown": [
                "Record the displayed alarm or error code.",
                "Keep the machine stopped.",
                "Review recent maintenance history.",
                "Follow the approved restart procedure.",
            ],
        },
        "safety": [
            "Keep clear of the mold closing area.",
            "Use lockout and tagout before maintenance inside hazardous areas.",
            "Do not touch hot surfaces or molten material.",
            "Never bypass machine safety interlocks.",
            "Follow emergency procedures for hydraulic leaks, fire, smoke, or electrical faults.",
        ],
    },

    "Robotic Welding Cell": {
        "manual": {
            "welding arc unstable": [
                "Stop the welding operation before inspection.",
                "Inspect accessible welding consumables.",
                "Check the maintenance log for previous arc stability issues.",
                "Follow the approved welding-system inspection procedure.",
            ],
            "robot movement error": [
                "Stop the robot before inspection.",
                "Record the controller alarm.",
                "Review recent maintenance records.",
                "Do not manually move the robot outside the approved procedure.",
            ],
            "wire feeding issue": [
                "Stop the welding operation.",
                "Inspect the accessible wire-feed path.",
                "Review maintenance records for previous wire-feed problems.",
                "Replace consumables according to the approved procedure.",
            ],
            "excessive vibration": [
                "Stop the robot before inspection.",
                "Inspect accessible tooling and fixtures.",
                "Review previous maintenance records.",
                "Do not enter the robot work envelope without authorization.",
            ],
            "controller alarm": [
                "Record the exact controller alarm.",
                "Keep the robot stopped.",
                "Review maintenance history for the alarm.",
                "Follow the approved controller recovery procedure.",
            ],
            "unexpected shutdown": [
                "Record the alarm or error code.",
                "Keep the welding cell stopped.",
                "Review recent maintenance records.",
                "Follow the approved restart procedure.",
            ],
        },
        "safety": [
            "Do not enter the robot work envelope while the robot is enabled.",
            "Use the approved lockout and tagout procedure before authorized maintenance.",
            "Follow welding fume and eye-protection requirements.",
            "Do not bypass robot safety interlocks.",
            "Follow emergency procedures for electrical faults, fire, or uncontrolled robot movement.",
        ],
    },

    "Air Compressor": {
        "manual": {
            "low air pressure": [
                "Check the pressure reading and record the observed value.",
                "Inspect accessible air lines for visible leakage.",
                "Review maintenance records for previous pressure problems.",
                "Check the compressor operating status according to the manual.",
            ],
            "high discharge temperature": [
                "Stop the compressor and allow it to cool.",
                "Check ventilation around the compressor.",
                "Review maintenance history for previous temperature problems.",
                "Do not access hot internal components while the compressor is operating.",
            ],
            "oil leak": [
                "Stop the compressor before inspection.",
                "Identify the visible location of the leak without contacting hot or pressurized components.",
                "Review maintenance records for previous oil leaks.",
                "Do not open pressurized components unless authorized.",
            ],
            "abnormal noise": [
                "Stop the compressor if abnormal noise increases.",
                "Record when the noise occurs.",
                "Inspect accessible external components.",
                "Review maintenance records for similar noise reports.",
            ],
            "compressor not starting": [
                "Check the control panel for alarms.",
                "Confirm the compressor is in the correct operating state.",
                "Review recent maintenance records.",
                "Follow the approved startup procedure.",
            ],
            "unexpected shutdown": [
                "Record the alarm or error code.",
                "Keep the compressor stopped.",
                "Review maintenance history.",
                "Follow the approved restart procedure.",
            ],
        },
        "safety": [
            "Stop and isolate the compressor before authorized maintenance.",
            "Do not open pressurized components while pressure remains in the system.",
            "Keep clear of hot surfaces and moving components.",
            "Use the approved lockout and tagout procedure.",
            "Follow emergency procedures for fire, electrical faults, or significant air-system leaks.",
        ],
    },
}


# ============================================================
# Error Codes
# ============================================================

ERROR_CODES = {
    "overheating": ["TEMP-HIGH", "TEMP-ALARM"],
    "excessive vibration": ["VIB-101", "VIB-205"],
    "coolant flow issue": ["COOL-LOW", "COOL-FLOW"],
    "spindle not starting": ["SPN-START", "SPN-401"],
    "unexpected shutdown": ["SYS-STOP", "PWR-FAIL"],
    "tool wear": ["TOOL-WEAR", "TOOL-207"],
    "poor surface finish": ["SURF-QUAL", "FIN-102"],
    "abnormal noise": ["NOISE-101", "NOISE-202"],
    "chuck problem": ["CHUCK-101", "CHUCK-LOCK"],
    "spindle overheating": ["SPN-TEMP", "TEMP-301"],
    "tool alignment issue": ["ALIGN-101", "TOOL-ALIGN"],
    "low pressure": ["PRESS-LOW", "HYD-101"],
    "pressure fluctuation": ["PRESS-FLUC", "HYD-205"],
    "hydraulic leak": ["HYD-LEAK", "HYD-301"],
    "slow operation": ["SPEED-LOW", "HYD-SLOW"],
    "drill bit slipping": ["DRILL-101", "CHUCK-SLIP"],
    "motor overheating": ["MOTOR-TEMP", "TEMP-401"],
    "poor drilling accuracy": ["DRILL-ACC", "ALIGN-205"],
    "spindle vibration": ["SPN-VIB", "VIB-301"],
    "coolant issue": ["COOL-101", "COOL-205"],
    "grinding wheel vibration": ["GRIND-VIB", "WHEEL-VIB"],
    "wheel wear": ["WHEEL-WEAR", "GRIND-201"],
    "belt slipping": ["BELT-SLIP", "CON-101"],
    "belt misalignment": ["BELT-ALIGN", "CON-205"],
    "conveyor not starting": ["CON-START", "MOTOR-START"],
    "injection pressure low": ["INJ-PRESS", "PRESS-401"],
    "material temperature high": ["MAT-TEMP", "TEMP-501"],
    "hydraulic pressure low": ["HYD-PRESS", "PRESS-LOW"],
    "poor molding quality": ["MOLD-QUAL", "MOLD-201"],
    "machine not starting": ["MACH-START", "START-101"],
    "welding arc unstable": ["ARC-101", "WELD-205"],
    "robot movement error": ["ROBOT-401", "MOTION-101"],
    "wire feeding issue": ["WIRE-101", "FEED-205"],
    "controller alarm": ["CTRL-401", "ROBOT-ALARM"],
    "low air pressure": ["AIR-LOW", "PRESS-501"],
    "high discharge temperature": ["AIR-TEMP", "TEMP-601"],
    "oil leak": ["OIL-LEAK", "COMP-301"],
    "compressor not starting": ["COMP-START", "START-501"],
}


# ============================================================
# Utility Functions
# ============================================================

def ensure_directories():
    """Create all seed-data directories."""

    MANUALS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MAINTENANCE_LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAFETY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def machine_filename(machine_name: str) -> str:
    """Convert a machine name into a safe filename."""

    return (
        machine_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def get_error_code(problem: str) -> str | None:
    """Return a controlled error code for a known problem."""

    codes = ERROR_CODES.get(problem)

    if not codes:
        return None

    return random.choice(codes)


# ============================================================
# Manual Generation
# ============================================================

def generate_manual(machine: dict):
    """Generate one controlled maintenance manual."""

    machine_name = machine["machine"]

    technical = TECHNICAL_DATA[
        machine_name
    ]["manual"]

    filename = (
        f"{machine_filename(machine_name)}_manual.txt"
    )

    file_path = MANUALS_DIR / filename

    lines = []

    lines.append(
        f"{machine_name} Maintenance Manual"
    )

    lines.append("=" * 50)

    lines.append("")

    lines.append(
        "Machine Information"
    )

    lines.append(
        "-------------------"
    )

    lines.append(
        f"Machine: {machine_name}"
    )

    lines.append(
        f"Machine ID: {machine['machine_id']}"
    )

    lines.append(
        f"Manufacturer: {machine['manufacturer']}"
    )

    lines.append(
        f"Model: {machine['model']}"
    )

    lines.append(
        f"Department: {machine['department']}"
    )

    lines.append(
        f"Location: {machine['location']}"
    )

    lines.append("")

    lines.append(
        "General Troubleshooting Guidance"
    )

    lines.append(
        "--------------------------------"
    )

    lines.append(
        "Use this manual together with approved "
        "maintenance logs and safety procedures."
    )

    lines.append(
        "Technicians must stop the machine before "
        "performing inspection or maintenance."
    )

    lines.append(
        "Do not perform procedures that are not "
        "supported by approved documentation."
    )

    lines.append("")

    for problem, steps in technical.items():

        lines.append(
            f"Problem: {problem}"
        )

        lines.append(
            "-" * (len(problem) + 9)
        )

        for number, step in enumerate(
            steps,
            start=1,
        ):

            lines.append(
                f"{number}. {step}"
            )

        lines.append("")

    lines.append(
        "Escalation Guidance"
    )

    lines.append(
        "-------------------"
    )

    lines.append(
        "If the documented troubleshooting steps "
        "do not resolve the issue, stop further "
        "intervention and escalate the problem "
        "to the authorized maintenance supervisor."
    )

    file_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return file_path


# ============================================================
# Safety Document Generation
# ============================================================

def generate_safety_document(machine: dict):
    """Generate one controlled safety document."""

    machine_name = machine["machine"]

    safety_rules = TECHNICAL_DATA[
        machine_name
    ]["safety"]

    filename = (
        f"{machine_filename(machine_name)}_safety.txt"
    )

    file_path = SAFETY_DIR / filename

    lines = []

    lines.append(
        f"{machine_name} Safety Procedures"
    )

    lines.append("=" * 50)

    lines.append("")

    lines.append(
        "Machine Information"
    )

    lines.append(
        "-------------------"
    )

    lines.append(
        f"Machine: {machine_name}"
    )

    lines.append(
        f"Machine ID: {machine['machine_id']}"
    )

    lines.append(
        f"Manufacturer: {machine['manufacturer']}"
    )

    lines.append(
        f"Model: {machine['model']}"
    )

    lines.append(
        f"Location: {machine['location']}"
    )

    lines.append("")

    lines.append(
        "Approved Safety Procedures"
    )

    lines.append(
        "--------------------------"
    )

    for number, rule in enumerate(
        safety_rules,
        start=1,
    ):

        lines.append(
            f"{number}. {rule}"
        )

    lines.append("")

    lines.append(
        "Emergency Escalation"
    )

    lines.append(
        "--------------------"
    )

    lines.append(
        "For fire, smoke, electrical sparking, "
        "serious injury, uncontrolled machine movement, "
        "or another immediate hazard, stop operation "
        "when it is safe to do so and follow the "
        "organization's approved emergency procedure."
    )

    lines.append("")

    lines.append(
        "Safety Note"
    )

    lines.append(
        "-----------"
    )

    lines.append(
        "Safety procedures in this document are "
        "machine-specific approved guidance. "
        "Do not substitute undocumented procedures."
    )

    file_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return file_path


# ============================================================
# Maintenance Log Generation
# ============================================================

def generate_maintenance_log(machine: dict):
    """
    Generate one maintenance log containing
    multiple realistic historical records.
    """

    machine_name = machine["machine"]

    filename = (
        f"{machine_filename(machine_name)}_maintenance.txt"
    )

    file_path = (
        MAINTENANCE_LOGS_DIR / filename
    )

    problems = machine["problems"]

    lines = []

    lines.append(
        f"{machine_name} Maintenance History"
    )

    lines.append("=" * 50)

    lines.append("")

    lines.append(
        f"Machine ID: {machine['machine_id']}"
    )

    lines.append(
        f"Manufacturer: {machine['manufacturer']}"
    )

    lines.append(
        f"Model: {machine['model']}"
    )

    lines.append(
        f"Department: {machine['department']}"
    )

    lines.append(
        f"Location: {machine['location']}"
    )

    lines.append("")

    base_date = (
        datetime.now()
        - timedelta(days=365)
    )

    for record_number in range(
        1,
        MAINTENANCE_RECORDS_PER_MACHINE + 1,
    ):

        problem = random.choice(
            problems
        )

        error_code = get_error_code(
            problem
        )

        technician = fake.name()

        work_order = (
            f"WO-{machine['machine_id']}-"
            f"{1000 + record_number}"
        )

        inspection_id = (
            f"INS-{fake.random_int(10000, 99999)}"
        )

        record_date = (
            base_date
            + timedelta(
                days=random.randint(
                    0,
                    365,
                )
            )
        )

        resolution_days = random.choice(
            [
                0,
                0,
                1,
                1,
                2,
                3,
                5,
            ]
        )

        if resolution_days == 0:

            status = (
                "Resolved during inspection"
            )

        elif resolution_days <= 2:

            status = "Resolved"

        else:

            status = (
                "Escalated and resolved"
            )

        lines.append(
            f"Maintenance Record {record_number}"
        )

        lines.append(
            "-" * 30
        )

        lines.append(
            "Date: "
            f"{record_date.strftime('%Y-%m-%d')}"
        )

        lines.append(
            f"Work Order: {work_order}"
        )

        lines.append(
            f"Inspection ID: {inspection_id}"
        )

        lines.append(
            f"Technician: {technician}"
        )

        lines.append(
            f"Problem: {problem}"
        )

        if error_code:

            lines.append(
                f"Error Code: {error_code}"
            )

        lines.append(
            f"Status: {status}"
        )

        lines.append(
            f"Resolution Time: "
            f"{resolution_days} day(s)"
        )

        lines.append(
            "Maintenance Note: "
            f"The technician recorded a {problem} "
            "condition and performed the documented "
            "inspection procedure."
        )

        lines.append("")

    file_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return file_path


# ============================================================
# Indexing
# ============================================================

def index_document(
    file_path: Path,
    document_type: str,
    machine: dict,
    owner: str,
    version: str = "1.0",
):
    """
    Chunk, embed, and index one document.

    If the document is already recorded as indexed,
    skip it so the seed process can safely resume.
    """

    existing_documents = get_documents(
        limit=1000
    )

    for document in existing_documents:

        if (
            document["filename"]
            == file_path.name
            and document["status"]
            == "indexed"
        ):

            print(
                f"Skipping: {file_path.name} "
                "(already indexed)"
            )

            return 0

    print(
        f"Indexing: {file_path.name}"
    )

    chunks = create_chunks(
        file_path=file_path,
        document_type=document_type,
        machine=machine["machine"],
        version=version,
        owner=owner,
    )

    if not chunks:

        print(
            f"WARNING: No readable content in "
            f"{file_path.name}"
        )

        return 0

    embeddings = create_embeddings(
        [
            chunk["text"]
            for chunk in chunks
        ],
        batch_size=20,
    )

    add_documents(
        chunks=chunks,
        embeddings=embeddings,
    )

    save_document(
        filename=file_path.name,
        document_type=document_type,
        machine=machine["machine"],
        version=version,
        owner=owner,
        chunks=len(chunks),
        status="indexed",
    )

    print(
        f"  -> {len(chunks)} chunks indexed"
    )

    return len(chunks)


# ============================================================
# Main Seed Process
# ============================================================

def main():
    """Generate and index the complete Neri seed dataset."""

    print("")

    print("=" * 60)
    print("NERI DATA SEEDING")
    print("=" * 60)

    print("")

    initialize_database()

    ensure_directories()

    total_documents = 0

    total_chunks = 0

    print(
        f"Machines: {len(MACHINES)}"
    )

    print(
        "Maintenance records per machine: "
        f"{MAINTENANCE_RECORDS_PER_MACHINE}"
    )

    print(
        "Expected maintenance records: "
        f"{len(MACHINES) * MAINTENANCE_RECORDS_PER_MACHINE}"
    )

    print("")

    # ========================================================
    # Generate and index documents
    # ========================================================

    for machine in MACHINES:

        machine_name = machine["machine"]

        print("")

        print(
            f"[{machine_name}]"
        )

        # ----------------------------------------------------
        # Manual
        # ----------------------------------------------------

        manual_path = generate_manual(
            machine
        )

        manual_chunks = index_document(
            file_path=manual_path,
            document_type="manual",
            machine=machine,
            owner=machine["manual_owner"],
        )

        total_chunks += manual_chunks

        if manual_chunks > 0:
            total_documents += 1

        # ----------------------------------------------------
        # Maintenance Log
        # ----------------------------------------------------

        maintenance_path = (
            generate_maintenance_log(
                machine
            )
        )

        maintenance_chunks = index_document(
            file_path=maintenance_path,
            document_type="maintenance_log",
            machine=machine,
            owner=machine["manual_owner"],
        )

        total_chunks += maintenance_chunks

        if maintenance_chunks > 0:
            total_documents += 1

        # ----------------------------------------------------
        # Safety Document
        # ----------------------------------------------------

        safety_path = (
            generate_safety_document(
                machine
            )
        )

        safety_chunks = index_document(
            file_path=safety_path,
            document_type="safety",
            machine=machine,
            owner=machine["safety_owner"],
        )

        total_chunks += safety_chunks

        if safety_chunks > 0:
            total_documents += 1

    # ========================================================
    # Final Summary
    # ========================================================

    print("")

    print("=" * 60)
    print("NERI DATA SEEDING COMPLETE")
    print("=" * 60)

    print("")

    print(
        f"Machines: {len(MACHINES)}"
    )

    print(
        f"Documents indexed during this run: "
        f"{total_documents}"
    )

    print(
        "Expected source documents: 30"
    )

    print(
        "Maintenance records created: "
        f"{len(MACHINES) * MAINTENANCE_RECORDS_PER_MACHINE}"
    )

    print(
        f"Chunks indexed during this run: "
        f"{total_chunks}"
    )

    print("")

    print("Data locations:")

    print(
        f"  Manuals: {MANUALS_DIR}"
    )

    print(
        f"  Maintenance logs: "
        f"{MAINTENANCE_LOGS_DIR}"
    )

    print(
        f"  Safety documents: {SAFETY_DIR}"
    )

    print("")

    print(
        "Neri knowledge base is ready."
    )

    print("")


if __name__ == "__main__":
    main()