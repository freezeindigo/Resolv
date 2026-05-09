# HYPOTHESIS LIBRARY — Domain to Hypothesis Agent Mappings

This file defines which hypothesis agents are spawned for each complaint domain in Tier 3 processing.

## Design Principles

1. Each domain has 2-4 hypothesis types
2. Hypothesis agents are NOT fixed — they're spawned dynamically based on domain + available evidence
3. Each hypothesis agent has isolated system prompt + evidence filter
4. Evidence filter ensures cognitive isolation — each agent only sees evidence relevant to its hypothesis

## Configuration Format

Each hypothesis is defined in `src/config/hypothesis_library.yaml`:

```yaml
domain_name:
  description: "Brief description"
  trigger_keywords: [...]          # For domain classifier
  hypothesis_types:
    - id: unique_hypothesis_id
      name: "Human-readable name"
      description: "What this hypothesis represents"
      prompt_template: "src/agents/prompts/{id}.md"
      evidence_filter:
        - flat_history                # Which context fields to include
        - adjacent_above              # Specific adjacency directions
        - building_category_pattern
        - seasonal_data
      cost_of_error_weight: 1.0       # Relative cost vs other hypotheses
```

## Domain: Water / Plumbing (36% of volume)

**Trigger keywords:** leak, seep, water, damp, moist, flood, overflow, drain, choke, flush, pipe, plumb, tap, geyser, WC, wash basin, sink

### Hypothesis: pipe_failure
- **Description:** Plumbing pipe leak, fixture failure, or drainage issue in this unit
- **Evidence filter:** flat_history (plumbing-specific), recent_plumbing_work_in_flat, pipe_age_if_known
- **Cost-of-error weight:** 1.0 (baseline — ₹500-1,500 per wrong dispatch)
- **System prompt emphasis:** "Consider ONLY evidence of plumbing failure in THIS unit. Ignore structural, environmental, and HVAC explanations entirely."

### Hypothesis: structural_seepage
- **Description:** Waterproofing degradation, wall/ceiling seepage, inter-unit water ingress
- **Evidence filter:** building_seepage_history, construction_age, monsoon_timing, wall_orientation (if known), vertical_stack_pattern, adjacent_above_flat_history
- **Cost-of-error weight:** 10.0 (₹5,000-50,000 per wrong dispatch — structural work)
- **System prompt emphasis:** "Evaluate structural failure: waterproofing, wall integrity, inter-unit water paths. Do NOT consider pipe-specific issues in THIS unit."

### Hypothesis: environmental
- **Description:** Weather-driven water ingress, drainage overflow, external sources
- **Evidence filter:** recent_weather_data, drainage_system_status, roof_inspection_history, season
- **Cost-of-error weight:** 2.0
- **System prompt emphasis:** "Evaluate external/environmental water sources only. Weather, drainage, external conditions."

### Hypothesis: hvac_condensate
- **Description:** AC condensate drain failure (for properties with central AC or heavy AC usage)
- **Evidence filter:** hvac_system_type, recent_ac_service_history, ac_complaint_correlation
- **Cost-of-error weight:** 1.5
- **Trigger condition:** Only spawned if property has AC and complaint is ceiling-related
- **System prompt emphasis:** "Evaluate whether this is AC condensate drainage failure. Not plumbing, not structural."

## Domain: Electrical (19.7% of volume)

**Trigger keywords:** electr, power, MCB, switch, socket, wir, light, fan, exhaust, trip, volt, meter, intercom, VDP, bell, door phone

### Hypothesis: internal_wiring
- **Description:** Wiring fault, fixture failure, or circuit issue within this unit
- **Evidence filter:** flat_history (electrical), recent_electrical_work, wiring_age
- **Cost-of-error weight:** 1.0

### Hypothesis: external_supply
- **Description:** Building-level electrical supply issue, load shedding, incoming power problem
- **Evidence filter:** building_electrical_complaints, power_cut_patterns, transformer_history
- **Cost-of-error weight:** 2.0

### Hypothesis: equipment_specific
- **Description:** Failure of specific equipment (geyser, fan, intercom, VDP)
- **Evidence filter:** equipment_age, warranty_status, recent_equipment_service
- **Cost-of-error weight:** 0.8

### Hypothesis: safety_hazard
- **Description:** Shock risk, short circuit, fire hazard requiring immediate action
- **Evidence filter:** keywords (shock, spark, burning smell, smoke), severity_indicators
- **Cost-of-error weight:** 20.0 (highest — life safety)
- **System prompt emphasis:** "Evaluate IMMEDIATE safety risk. If likelihood is >0.3, escalate regardless of other hypotheses."

## Domain: Structural / Civil (9.2% of volume)

**Trigger keywords:** crack, wall, ceil, tile, paint, plaster, grout, concrete, structural, civil, mason, brick

### Hypothesis: settlement_movement
- **Description:** Building settlement, thermal movement, or age-related structural shift
- **Evidence filter:** building_age, settlement_complaint_history, crack_location_pattern
- **Cost-of-error weight:** 5.0

### Hypothesis: waterproofing_failure
- **Description:** Failed waterproofing causing structural damage (cracks + moisture)
- **Evidence filter:** monsoon_history, exterior_wall_complaints, moisture_indicators
- **Cost-of-error weight:** 10.0

### Hypothesis: installation_defect
- **Description:** Original construction defect, tile de-bonding, grout failure
- **Evidence filter:** dlp_status, original_construction_date, tile_age, warranty_status
- **Cost-of-error weight:** 3.0 (may be developer responsibility)

### Hypothesis: environmental_damage
- **Description:** Damage from external factors (storm, earthquake, flooding)
- **Evidence filter:** recent_weather_events, seismic_activity, flood_events
- **Cost-of-error weight:** 2.0

## Domain: Carpentry (15.2% of volume)

**Trigger keywords:** door, window, lock, handle, hinge, cabinet, wardrobe, shutter, sliding, frame, wood, carpenter

### Hypothesis: mechanical_wear
- **Description:** Normal wear-and-tear on moving parts (locks, hinges, handles)
- **Evidence filter:** installation_age, usage_pattern_if_known
- **Cost-of-error weight:** 1.0

### Hypothesis: structural_movement
- **Description:** Door/window misalignment due to building settlement or frame shift
- **Evidence filter:** building_age, other_carpentry_complaints_same_unit, settlement_indicators
- **Cost-of-error weight:** 4.0 (if structural, wrong fix won't last)

### Hypothesis: installation_defect
- **Description:** Original installation issue (common in DLP period)
- **Evidence filter:** dlp_status, original_fitout_date
- **Cost-of-error weight:** 2.0

### Hypothesis: environmental_damage
- **Description:** Humidity warping, monsoon-related swelling
- **Evidence filter:** season, humidity_data, multiple_unit_pattern
- **Cost-of-error weight:** 1.5

## Domain: HVAC (3.3% of volume)

**Trigger keywords:** AC, A/C, air cond, cool, heat, HVAC, compressor, gas leak, refrigerant

### Hypothesis: compressor_gas
- **Description:** Compressor failure or refrigerant gas issue
- **Evidence filter:** ac_age, recent_service_history, cooling_pattern_description
- **Cost-of-error weight:** 2.0

### Hypothesis: electrical_fault
- **Description:** Electrical issue affecting AC (voltage, wiring, PCB)
- **Evidence filter:** electrical_complaints_in_building, recent_power_fluctuations
- **Cost-of-error weight:** 1.5

### Hypothesis: post_service_issue
- **Description:** AC stopped working correctly after recent service — vendor issue
- **Evidence filter:** recent_ac_service_date, vendor_id_if_serviced, service_complaints_for_vendor
- **Cost-of-error weight:** 3.0 (vendor accountability)

### Hypothesis: external_factor
- **Description:** External factors (heat wave, insufficient cooling capacity for ambient)
- **Evidence filter:** outdoor_temperature, building_wide_cooling_issues
- **Cost-of-error weight:** 1.0

## Domain: Lift / Elevator (2.8% of volume)

**Trigger keywords:** lift, elevator, escalator

### Hypothesis: electrical_fault
- **Evidence filter:** lift_maintenance_history, recent_electrical_issues_in_building
- **Cost-of-error weight:** 5.0 (safety concern)

### Hypothesis: mechanical_wear
- **Evidence filter:** lift_age, last_major_service, usage_volume
- **Cost-of-error weight:** 3.0

### Hypothesis: overload_misuse
- **Evidence filter:** recent_heavy_material_movement, resident_complaints_about_usage
- **Cost-of-error weight:** 1.0

### Hypothesis: maintenance_overdue
- **Evidence filter:** amc_contract_status, last_service_date, service_frequency
- **Cost-of-error weight:** 2.0

## Domain: Common Area (10.5% of volume)

**Special handling:** Common area complaints often inherit hypotheses from their underlying domain (water/electrical/structural). The domain classifier identifies the primary underlying domain, and hypothesis agents from THAT domain are spawned with a "common area" flag that modifies evidence filters (building-level data weighted higher than unit-level).

## Domain: Safety / Security (3.5% of volume)

**Trigger keywords:** fire, smoke, gas, safety, danger, hazard, emergency, security, guard, CCTV, boom barrier

### Hypothesis: fire_system
- **Cost-of-error weight:** 50.0 (life safety)

### Hypothesis: gas_leak
- **Cost-of-error weight:** 50.0 (life safety)

### Hypothesis: structural_hazard
- **Cost-of-error weight:** 20.0

### Hypothesis: security_equipment
- **Cost-of-error weight:** 3.0

**For safety domain, ALL hypothesis agents are spawned regardless of tier. Even if primary hypothesis is low-confidence, high-cost safety hypotheses trigger escalation.**

## Domain: Pest / Hygiene (7.7% of volume)

**Trigger keywords:** pest, mosquit, cockroach, rat, rodent, lizard, insect, clean, garbage, smell, odour, stink, hygiene

Simpler domain — usually Tier 1 or Tier 2 unless there's a root cause (e.g., smell from plumbing, pests from structural gap). If root cause suspected, classifier promotes to the primary domain (water_plumbing or structural_civil) instead.

## Adding New Domains (Platform Extensibility)

To add a new domain (e.g., `commercial_bms` for commercial properties):

1. Add entry to `src/config/hypothesis_library.yaml`
2. Define trigger keywords
3. List hypothesis types with evidence filters
4. Create prompt templates in `src/agents/prompts/`
5. Add domain-specific evidence fields to Context Assembler if needed

The orchestration layer does not change. This is the core extensibility property.
