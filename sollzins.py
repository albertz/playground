
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Name des Kredits, der Bank, etc", required=True)
parser.add_argument("--kredit", type=float, help="Kreditbetrag in Euro", required=True)
parser.add_argument("--sollzins", type=float, help="Sollzins in Prozent", required=True)
parser.add_argument("--rate", type=float, help="Monatliche Rate in Euro", required=True)
parser.add_argument("--laufzeit", type=int, help="Laufzeit in Monaten", required=True)
parser.add_argument(
    "--target-laufzeit", type=int, required=False,
    help="gewünschte Laufzeit in Monaten (optional, weniger durch Tilgung pro Monat)")
args = parser.parse_args()

rest_kredit = args.kredit
sollzins_year = args.sollzins / 100
effektiver_zins_year = (1 + sollzins_year / 12) ** 12 - 1
rate_month = args.rate
print("Name:", args.name)
print(f"Start: {rest_kredit:.2f} Euro")
print(f"Laufzeit: {args.laufzeit // 12} Jahre, {args.laufzeit % 12} Monate")
print(f"Monatliche Rate: {rate_month:.2f} Euro")
print(f"Sollzins: {sollzins_year * 100:.2f} % p.a.")
print(f"Effektiver Zins: {effektiver_zins_year * 100:.2f} % p.a.")


def iter_months(*, extra_tilgung: float = 0.0, laufzeit=args.laufzeit, file=sys.stdout, exit_on_zero=False):
    rest_kredit = args.kredit
    for month in range(laufzeit):
        rest_kredit += rest_kredit * sollzins_year / 12
        rest_kredit -= rate_month
        rest_kredit -= extra_tilgung

        if month < 3 or month % 12 == 0 or rest_kredit < 10_000:
            year = month // 12
            print(f"Jahr {year}, Ende Monat {month % 12 + 1}: {rest_kredit:.2f} Euro", file=file)
        if rest_kredit <= 0 and exit_on_zero:
            return month + 1
    print(f"Ende: {rest_kredit:.2f} Euro", file=file)


iter_months()


def calc_num_months(*, extra_tilgung: float = 0.0):
    n = iter_months(extra_tilgung=extra_tilgung, exit_on_zero=True, file=open("/dev/null", "w"))
    assert n is not None
    return n


if args.target_laufzeit:
    assert args.target_laufzeit <= args.laufzeit
    target_num_months = args.target_laufzeit
    print(f"Gewünschte Laufzeit: {target_num_months // 12} Jahre, {target_num_months % 12} Monate")
    print("Berechne zusätzliche monatige Tilgung...")
    target_extra_tilgung = 0.0
    while True:
        num_months = calc_num_months(extra_tilgung=target_extra_tilgung)
        if num_months <= target_num_months:
            break
        target_extra_tilgung += 10

    iter_months(extra_tilgung=target_extra_tilgung, laufzeit=target_num_months)
    print(f"Zusätzliche monatige Tilgung: {target_extra_tilgung:.2f} Euro")
    print(f"Gesamt monatliche Rate: {rate_month + target_extra_tilgung:.2f} Euro")


print("Name:", args.name)
