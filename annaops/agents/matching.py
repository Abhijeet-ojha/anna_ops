from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

from annaops.config import Settings, settings
from annaops.schemas import ExtractedOffer, MatchCandidate, MatchResult, RouteEstimate


FICTIONAL_NGOS: list[dict[str, Any]] = [
    {"id": "ngo-01", "name": "Saffron Bridge Collective", "service_area": "Jaipur Central (synthetic)", "latitude": 26.9124, "longitude": 75.7873, "accepted_categories": ["vegetarian", "mixed"], "max_capacity": 220, "available_capacity": 180, "opens_at": "08:00", "closes_at": "22:00", "available": True, "minimum_quantity": 10},
    {"id": "ngo-02", "name": "Blue Bowl Network", "service_area": "Jaipur North (synthetic)", "latitude": 26.9500, "longitude": 75.8000, "accepted_categories": ["vegetarian", "non_vegetarian", "mixed", "unknown"], "max_capacity": 150, "available_capacity": 95, "opens_at": "00:00", "closes_at": "23:59", "available": True, "minimum_quantity": 5},
    {"id": "ngo-03", "name": "Kindred Plate Foundation", "service_area": "Jaipur West (synthetic)", "latitude": 26.9050, "longitude": 75.7400, "accepted_categories": ["vegetarian"], "max_capacity": 100, "available_capacity": 72, "opens_at": "07:00", "closes_at": "21:30", "available": True, "minimum_quantity": 8},
    {"id": "ngo-04", "name": "Harvest Halo Trust", "service_area": "Jaipur South (synthetic)", "latitude": 26.8500, "longitude": 75.7900, "accepted_categories": ["vegetarian", "mixed"], "max_capacity": 300, "available_capacity": 240, "opens_at": "06:00", "closes_at": "20:00", "available": True, "minimum_quantity": 20},
    {"id": "ngo-05", "name": "Lantern Lunch Mission", "service_area": "Jaipur East (synthetic)", "latitude": 26.9100, "longitude": 75.8500, "accepted_categories": ["vegetarian", "non_vegetarian", "mixed"], "max_capacity": 80, "available_capacity": 55, "opens_at": "10:00", "closes_at": "22:30", "available": True, "minimum_quantity": 5},
    {"id": "ngo-06", "name": "Mosaic Meals Alliance", "service_area": "Sanganer (synthetic)", "latitude": 26.8200, "longitude": 75.8000, "accepted_categories": ["vegetarian", "non_vegetarian", "mixed", "unknown"], "max_capacity": 180, "available_capacity": 120, "opens_at": "00:00", "closes_at": "23:59", "available": True, "minimum_quantity": 12},
    {"id": "ngo-07", "name": "Open Palm Pantry", "service_area": "Amer (synthetic)", "latitude": 26.9850, "longitude": 75.8500, "accepted_categories": ["vegetarian", "unknown"], "max_capacity": 90, "available_capacity": 60, "opens_at": "07:30", "closes_at": "20:30", "available": True, "minimum_quantity": 5},
    {"id": "ngo-08", "name": "Silver Grain Society", "service_area": "Jagatpura (synthetic)", "latitude": 26.8350, "longitude": 75.8650, "accepted_categories": ["vegetarian", "non_vegetarian", "mixed"], "max_capacity": 200, "available_capacity": 0, "opens_at": "08:00", "closes_at": "22:00", "available": False, "minimum_quantity": 15},
    {"id": "ngo-09", "name": "Sunrise Serving Circle", "service_area": "Vidhyadhar Nagar (synthetic)", "latitude": 26.9650, "longitude": 75.7800, "accepted_categories": ["vegetarian", "mixed"], "max_capacity": 140, "available_capacity": 110, "opens_at": "05:30", "closes_at": "19:00", "available": True, "minimum_quantity": 10},
    {"id": "ngo-10", "name": "Together Table Guild", "service_area": "Mansarovar (synthetic)", "latitude": 26.8550, "longitude": 75.7650, "accepted_categories": ["vegetarian", "non_vegetarian", "mixed", "unknown"], "max_capacity": 260, "available_capacity": 200, "opens_at": "00:00", "closes_at": "23:59", "available": True, "minimum_quantity": 20},
    {"id": "ngo-11", "name": "Willow Basket Initiative", "service_area": "Civil Lines (synthetic)", "latitude": 26.9050, "longitude": 75.7750, "accepted_categories": ["vegetarian"], "max_capacity": 65, "available_capacity": 35, "opens_at": "08:00", "closes_at": "21:00", "available": True, "minimum_quantity": 5},
    {"id": "ngo-12", "name": "Yellow Apron Community", "service_area": "Malviya Nagar (synthetic)", "latitude": 26.8600, "longitude": 75.8150, "accepted_categories": ["vegetarian", "non_vegetarian", "mixed"], "max_capacity": 175, "available_capacity": 130, "opens_at": "07:00", "closes_at": "22:00", "available": True, "minimum_quantity": 10},
]

SYNTHETIC_DEMANDS: dict[str, list[dict[str, Any]]] = {
    "ngo-01": [{"id": "demand-ngo-01-meals", "item_name": "vegetarian food", "requested_quantity": 100, "unit": "portions", "minimum_quantity": 10, "priority": 70}],
    "ngo-02": [
        {"id": "demand-ngo-02-rajma", "item_name": "rajma chawal", "requested_quantity": 50, "unit": "kg", "minimum_quantity": 20, "priority": 90},
        {"id": "demand-ngo-02-chicken-rice", "item_name": "chicken rice", "requested_quantity": 200, "unit": "portions", "minimum_quantity": 10, "priority": 10},
    ],
    "ngo-03": [{"id": "demand-ngo-03-nachos", "item_name": "nachos", "requested_quantity": 50, "unit": "kg", "minimum_quantity": 20, "priority": 60}],
    "ngo-04": [
        {"id": "demand-ngo-04-rice", "item_name": "cooked rice", "requested_quantity": 30, "unit": "kg", "minimum_quantity": 15, "priority": 80},
        {"id": "demand-ngo-04-dal", "item_name": "dal", "requested_quantity": 20, "unit": "kg", "minimum_quantity": 10, "priority": 80},
    ],
    "ngo-05": [{"id": "demand-ngo-05-paneer", "item_name": "paneer curry", "requested_quantity": 25, "unit": "kg", "minimum_quantity": 10, "priority": 70}],
    "ngo-06": [
        {"id": "demand-ngo-06-meals", "item_name": "vegetarian food", "requested_quantity": 100, "unit": "portions", "minimum_quantity": 25, "priority": 100},
        {"id": "demand-ngo-06-chicken-rice", "item_name": "chicken rice", "requested_quantity": 120, "unit": "portions", "minimum_quantity": 20, "priority": 10},
    ],
    "ngo-07": [{"id": "demand-ngo-07-pasta", "item_name": "pasta", "requested_quantity": 40, "unit": "kg", "minimum_quantity": 20, "priority": 60}],
    "ngo-08": [{"id": "demand-ngo-08-rice", "item_name": "rice", "requested_quantity": 60, "unit": "kg", "minimum_quantity": 20, "priority": 50}],
    "ngo-09": [{"id": "demand-ngo-09-pasta", "item_name": "pasta", "requested_quantity": 50, "unit": "kg", "minimum_quantity": 25, "priority": 50}],
    "ngo-10": [{"id": "demand-ngo-10-rajma", "item_name": "rajma chawal", "requested_quantity": 50, "unit": "kg", "minimum_quantity": 20, "priority": 80}],
    "ngo-11": [{"id": "demand-ngo-11-rice", "item_name": "cooked rice", "requested_quantity": 20, "unit": "kg", "minimum_quantity": 10, "priority": 50}],
    "ngo-12": [{"id": "demand-ngo-12-dal", "item_name": "dal", "requested_quantity": 30, "unit": "kg", "minimum_quantity": 15, "priority": 50}],
}

for _profile in FICTIONAL_NGOS:
    _profile["demands"] = SYNTHETIC_DEMANDS.get(_profile["id"], [])


class MatchRouteAgent:
    def __init__(self, config: Settings = settings):
        self.config = config

    def match(self, offer: ExtractedOffer, ngos: list[dict[str, Any]], excluded_ids: set[str] | None = None,
              now: datetime | None = None) -> MatchResult:
        excluded_ids = excluded_ids or set()
        now = now or datetime.now(self.config.timezone)
        exclusions: dict[str, list[str]] = {}
        candidates: list[MatchCandidate] = []
        quantity = offer.quantity or 0
        if offer.food_items and len({item.unit for item in offer.food_items}) == 1:
            quantity = sum(item.quantity for item in offer.food_items)

        for ngo in ngos:
            reasons: list[str] = []
            matched_demands: list[dict[str, Any]] = []
            if ngo["id"] in excluded_ids:
                reasons.append("Already attempted for this donation.")
            if not ngo["available"]:
                reasons.append("Receiver is marked unavailable.")
            if ngo["available_capacity"] < quantity:
                reasons.append(f"Available capacity {ngo['available_capacity']:g} is below requested {quantity:g}.")
            if quantity < ngo["minimum_quantity"]:
                reasons.append(f"Quantity is below the receiver minimum of {ngo['minimum_quantity']:g}.")
            if offer.food_category not in ngo["accepted_categories"]:
                reasons.append(f"Food category '{offer.food_category}' is not accepted.")
            demands = [demand for demand in ngo.get("demands", []) if demand.get("active", True)]
            active_demands = []
            for demand in demands:
                deadline = demand.get("request_deadline")
                if deadline and datetime.fromisoformat(deadline.replace("Z", "+00:00")) <= now:
                    continue
                active_demands.append(demand)
            demands = active_demands
            for item in offer.food_items or [{
                "name": offer.food_type or "",
                "quantity": quantity,
                "unit": offer.unit or "",
                "category": offer.food_category,
            }]:
                item_name = item.name if hasattr(item, "name") else item["name"]
                item_quantity = item.quantity if hasattr(item, "quantity") else item["quantity"]
                item_unit = item.unit if hasattr(item, "unit") else item["unit"]
                item_category = item.category if hasattr(item, "category") else item["category"]
                demand = next(
                    (
                        candidate for candidate in demands
                        if candidate.get("remaining_quantity", candidate["requested_quantity"]) >= item_quantity
                        and candidate["minimum_quantity"] <= item_quantity
                        and candidate["unit"].lower() == item_unit.lower()
                        and (
                            candidate["item_name"].lower() == item_name.lower()
                            or (candidate["item_name"].lower() == "vegetarian food" and item_category == "vegetarian")
                        )
                    ),
                    None,
                )
                if demand is None:
                    reasons.append(f"No active demand accepts {item_quantity:g} {item_unit} of '{item_name}'.")
                else:
                    matched_demands.append({
                        "id": demand["id"], "item_name": demand["item_name"],
                        "requested_quantity": demand["requested_quantity"],
                        "remaining_quantity": demand.get("remaining_quantity", demand["requested_quantity"]),
                        "unit": demand["unit"], "minimum_quantity": demand["minimum_quantity"],
                    })
            if offer.pickup_deadline:
                local_deadline = offer.pickup_deadline.astimezone(self.config.timezone)
                deadline_minutes = local_deadline.hour * 60 + local_deadline.minute
                open_hour, open_minute = map(int, ngo["opens_at"].split(":"))
                close_hour, close_minute = map(int, ngo["closes_at"].split(":"))
                if not (open_hour * 60 + open_minute <= deadline_minutes <= close_hour * 60 + close_minute):
                    reasons.append("Pickup deadline falls outside receiver operating hours.")
            if reasons:
                exclusions[ngo["id"]] = reasons
                continue

            route = self._route(offer, ngo)
            if route.distance_km is None:
                score = 40.0
                score_reasons = ["Route unavailable; neutral distance score used."]
            else:
                distance_score = max(0.0, 50.0 - route.distance_km * 2.5)
                score = distance_score
                score_reasons = [f"Distance component {distance_score:.1f}/50 using {route.method}."]
            spare_ratio = min(1.0, (ngo["available_capacity"] - quantity) / max(1.0, ngo["available_capacity"]))
            capacity_score = 30.0 * (1.0 - spare_ratio * 0.35)
            time_score = 20.0
            score += capacity_score + time_score
            score_reasons.extend([
                f"Capacity component {capacity_score:.1f}/30; receiver can accept all {quantity:g} {offer.unit or 'units'}.",
                "Operating-hours component 20/20; stated deadline is within the receiving window.",
            ])
            candidates.append(MatchCandidate(
                ngo_id=ngo["id"], ngo_name=ngo["name"], score=round(score, 2),
                available_capacity=ngo["available_capacity"], reasons=score_reasons, route=route,
                matched_demands=matched_demands,
            ))

        candidates.sort(key=lambda item: (-item.score, item.ngo_id))
        for index, candidate in enumerate(candidates, 1):
            candidate.rank = index
        return MatchResult(
            candidates=candidates,
            exclusions=exclusions,
            rule_version=self.config.matching_rule_version,
            generated_at=now,
        )

    def _route(self, offer: ExtractedOffer, ngo: dict[str, Any]) -> RouteEstimate:
        if offer.donor_latitude is None or offer.donor_longitude is None:
            return RouteEstimate(
                distance_km=None, duration_minutes=None, method="unavailable", verified_road_route=False,
                note="Donor coordinates were not supplied; route feasibility requires human confirmation.",
            )
        if self.config.routing_base_url:
            try:
                url = (
                    f"{self.config.routing_base_url}/route/v1/driving/"
                    f"{offer.donor_longitude},{offer.donor_latitude};{ngo['longitude']},{ngo['latitude']}"
                    "?overview=false"
                )
                with urllib.request.urlopen(url, timeout=8) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                route = payload["routes"][0]
                return RouteEstimate(
                    distance_km=round(route["distance"] / 1000, 2),
                    duration_minutes=round(route["duration"] / 60, 1), method="osrm",
                    verified_road_route=True, note="Road estimate returned by the configured OSRM service.",
                )
            except (OSError, KeyError, IndexError, ValueError, json.JSONDecodeError, urllib.error.URLError):
                pass
        distance = self._haversine(
            offer.donor_latitude, offer.donor_longitude, ngo["latitude"], ngo["longitude"]
        )
        return RouteEstimate(
            distance_km=round(distance, 2), duration_minutes=None, method="straight_line_fallback",
            verified_road_route=False,
            note="Unverified straight-line distance; this is not a road distance or ETA.",
        )

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        delta_p, delta_l = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        value = math.sin(delta_p / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(delta_l / 2) ** 2
        return 2 * radius * math.atan2(math.sqrt(value), math.sqrt(1 - value))
