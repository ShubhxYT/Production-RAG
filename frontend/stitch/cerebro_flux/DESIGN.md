# Design System Document: The Intellectual Sanctuary
 
## 1. Overview & Creative North Star: "The Digital Curator"
This design system moves away from the chaotic, high-density look of traditional "enterprise" tools. Our Creative North Star is **The Digital Curator**. We are not building a database; we are building an intelligent, tranquil space where information is thoughtfully organized and surfaced.
 
The aesthetic is defined by **Soft Minimalism**. We replace rigid, boxy structures with fluid, expansive layouts. By leveraging large radii and intentional asymmetry—such as placing primary search elements off-center or using wide, breathable margins—we create an editorial experience that feels premium and human-centric. The "template" look is avoided by treating every screen as a physical landscape where depth is communicated through light and layering, not lines and borders.
 
## 2. Colors: Tonal Architecture
The palette is rooted in a calm, academic spectrum of blues and grays, punctuated by a singular, authoritative indigo.
The current color mode is **dark**.
 
### The "No-Line" Rule
**Strict Mandate:** 1px solid borders are prohibited for sectioning or containment. 
Boundaries must be defined exclusively through background color shifts. For example, a `surface-container-low` section should sit directly against a `surface` background. If you feel the need for a line, increase the whitespace instead.
 
### Surface Hierarchy & Nesting
Treat the UI as a series of stacked, fine-paper sheets. Use tiers to define importance:
- **Base Layer:** `surface` (#f7f9fb)
- **Secondary Sections:** `surface-container-low` (#f2f4f6)
- **Primary Content Cards:** `surface-container-lowest` (#ffffff) for maximum "lift."
- **Active Overlays:** `surface-bright` (#f7f9fb)
 
### The Glass & Gradient Rule
For floating elements (modals, dropdowns, or hovering citations), use **Glassmorphism**. Apply a semi-transparent `surface-container-lowest` (80% opacity) with a `backdrop-filter: blur(20px)`. 
 
For primary CTAs, do not use flat colors. Apply a subtle linear gradient from `primary` (#0030af) to `primary_container` (#0042e7) at a 135-degree angle to give the action "visual soul" and depth.
 
## 3. Typography: Editorial Authority
We utilize a dual-typeface system to balance professional weight with modern readability.
 
*   **Display & Headlines (Manrope):** Chosen for its geometric precision and modern warmth. Use `display-lg` and `headline-md` with slightly tighter letter-spacing (-0.02em) to create an authoritative, editorial feel.
*   **Body & Labels (Inter):** The workhorse for the RAG experience. Inter’s tall x-height ensures that complex AI-generated citations remain legible at `body-sm` scales.
 
**Hierarchy as Identity:** 
Large `display-sm` titles should be used to introduce search results, while `label-md` is reserved for metadata like "Source Confidence" or "Date Indexed." This contrast creates a rhythm that guides the eye through dense information.
 
## 4. Elevation & Depth: Tonal Layering
Traditional shadows are often "dirty." In this system, we use light and tone to simulate height.
 
*   **The Layering Principle:** Avoid elevation tokens. Instead, place a `surface-container-lowest` card on top of a `surface-container-low` background. This creates a natural "step-up" without visual noise.
*   **Ambient Shadows:** If a floating state (like a citation tooltip) is required, use a highly diffused shadow: `box-shadow: 0 20px 40px rgba(25, 28, 30, 0.06)`. The color is a tinted version of `on-surface`, never pure black.
*   **The "Ghost Border" Fallback:** If a container requires more definition for accessibility (e.g., in a search input), use a "Ghost Border": `outline-variant` (#c3c6d5) at **15% opacity**. This provides a hint of a boundary without breaking the soft aesthetic.
 
## 5. Components
 
### Search Inputs (The Hero)
The central RAG interface. 
- **Style:** Large radius (`xl`: 3rem), `surface-container-lowest` background. 
- **Interaction:** On focus, do not use a heavy stroke. Instead, transition the `box-shadow` to a deeper ambient glow and shift the background to a pure white.
 
### Document Cards & Citation Elements
- **Forbid Dividers:** Use `1.5rem` (md) vertical spacing to separate sources.
- **Citations:** Use `secondary_container` with a `lg` (2rem) radius. They should appear as "pills" embedded in the text.
- **Source Cards:** Use `surface-container-low` with a `md` radius. Ensure no borders are present; use `title-sm` for the document name and `body-sm` for the snippet.
 
### Buttons & Chips
- **Primary Button:** Indigo gradient (`primary` to `primary_container`), `full` radius, `label-md` text in uppercase with 0.05em tracking for a premium feel.
- **Action Chips:** Use `secondary_fixed_dim` with `on_secondary_fixed_variant` text. These should feel like tactile, rounded pebbles.
 
### AI Feedback "Pulse"
For RAG loading states, use a shimmering gradient across a `surface-container-highest` skeleton, moving from `primary_fixed` to `secondary_fixed`.
 
## 6. Do's and Don'ts
 
### Do
- **Do** use whitespace as a structural element. If an interface feels cluttered, increase the padding, don't add a line.
- **Do** use the `lg` (2rem) and `xl` (3rem) corner radii for large containers to maintain the "approachable" brand voice.
- **Do** ensure the `tertiary` teal color is used sparingly—only for "Success" states or verified source badges.
 
### Don't
- **Don't** use 100% black (#000000). Always use `on_surface` (#191c1e) for text to maintain a soft, premium contrast.
- **Don't** use "Drop Shadows" from standard UI kits. Stick to the Ambient Shadow values (low opacity, high blur).
- **Don't** stack more than three layers of surface containers. It leads to visual "muddiness." Use a maximum of: Base > Container > Card.