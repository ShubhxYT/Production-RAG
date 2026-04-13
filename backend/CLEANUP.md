# FullRag Project Cleanup — Before Frontend Development

This document outlines all cleanup tasks needed to prepare the repository for frontend development with React.

---

## **1. Remove Temporary Files**

- **`staging/`** — Delete temporary JSON files (used during ingestion testing)
  ```bash
  rm -rf staging/
  ```

- **`results/` folder** — Remove evaluation/experiment results if no longer needed
  ```bash
  rm -rf results/
  ```

- **`result/`** — Check if duplicate of `results/` and remove
  ```bash
  rm -rf result/
  ```

---

## **2. Archive Completed Backend Plans**

Create an archive directory and move completed plans:

```bash
mkdir -p plans/archive
mv plans/vector-database-setup/ plans/archive/
mv plans/embedding-service/ plans/archive/
mv plans/llm-generation-pipeline/ plans/archive/
mv plans/transcript-fallback/ plans/archive/
# Move any other completed feature plans to archive/
```

**Keep in `plans/` only:**
- Active/in-progress features
- Features still under development

---

## **3. Reorganize Test Structure**

Move backend tests into a clearer directory:

```bash
mkdir -p tests/backend
mv test/* tests/backend/
rmdir test/
```

This separates backend tests from frontend tests (to be added later as `tests/frontend/`).

---

## **4. Clean Up Root Files**

Move miscellaneous files to appropriate locations:

```bash
# Move cuda test to root tests folder (or archive)
mv cuda-test.py tests/backend/  # or rm cuda-test.py if not needed

# Move config files to deployment/config
mkdir -p deployment/config
mv pgvector.yaml deployment/config/
mv alembic.ini alembic/  # Already uses alembic config

# Organize sample data
mkdir -p data/samples
# (if needed, organize sample CSV/DJVU files)
```

---

## **5. Create Frontend Directory Structure**

```bash
mkdir -p frontend
cd frontend

# Create the React app structure
mkdir -p public
mkdir -p src/{components,pages,services,hooks,utils,styles}
```

**Frontend structure:**
```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/       # Reusable React components
│   ├── pages/            # Page-level components (layout)
│   ├── services/         # API calls to backend
│   ├── hooks/            # Custom React hooks
│   ├── utils/            # Helper functions
│   ├── styles/           # Global CSS/SCSS
│   ├── App.tsx
│   └── index.tsx
├── package.json
├── tsconfig.json
├── vite.config.ts        # or webpack.config.js
└── README.md
```

---

## **6. Update Backend Configuration**

**Option A: Remove Streamlit (if not needed)**
```bash
rm -rf streamlit_app/
```

**Option B: Keep Streamlit as optional dashboard**
- Leave `streamlit_app/` as-is
- Document it as optional in README

**Ensure API is clean:**
- Check `api/main.py` — should have proper CORS headers for React frontend
- Ensure all endpoints are documented
- Update API documentation

---

## **7. Update .gitignore**

Add or update `.gitignore` with these entries:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environments
venv/
ENV/
env/
.venv

# Testing
.pytest_cache/
.coverage
htmlcov/

# Environment files
.env
.env.local
.env.*.local

# Frontend
frontend/node_modules/
frontend/dist/
frontend/build/
frontend/.env.local
frontend/.env.development.local
frontend/.env.test.local
frontend/.env.production.local
frontend/.next/
frontend/out/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Misc
staging/
results/
```

---

## **8. Update Documentation**

### **Create `FRONTEND_SETUP.md`**
```markdown
# Frontend Development Setup

## Prerequisites
- Node.js 18+ or Bun
- npm or yarn or pnpm

## Installation
\`\`\`bash
cd frontend
npm install
\`\`\`

## Environment Variables
Create `.env.local`:
\`\`\`
VITE_API_URL=http://localhost:8001
\`\`\`

## Development
\`\`\`bash
npm run dev
\`\`\`

## Build
\`\`\`bash
npm run build
\`\`\`
```

### **Update Main `README.md`**

Add a section:
```markdown
## Project Structure

### Backend
- `api/` — FastAPI application
- `database/` — Database models & migrations
- `embeddings/` — Embedding service
- `generation/` — LLM generation & prompts
- `retrieval/` — RAG retrieval logic
- `ingestion/` — Document ingestion pipeline
- `evaluation/` — Evaluation framework
- `config/` — Configuration management
- `tests/backend/` — Backend tests

### Frontend
- `frontend/` — React application (coming soon)
- `frontend/src/components/` — React components
- `frontend/src/pages/` — Page layouts
- `frontend/src/services/` — API client
- `frontend/tests/` — Frontend/E2E tests

### Configuration
- `alembic/` — Database migrations
- `deployment/` — Deployment configs
```

---

## **9. Cleanup Checklist**

- [ ] Remove `staging/` directory
- [ ] Remove `results/` and/or `result/` directories
- [ ] Archive completed plans to `plans/archive/`
- [ ] Reorganize tests to `tests/backend/`
- [ ] Move/archive temporary files (`cuda-test.py`, etc.)
- [ ] Move config files to appropriate locations
- [ ] Create `frontend/` directory with React structure
- [ ] Decide on Streamlit (keep or remove)
- [ ] Update `.gitignore`
- [ ] Create `FRONTEND_SETUP.md`
- [ ] Update main `README.md`
- [ ] Verify API CORS headers are set correctly
- [ ] Run `git status` and commit cleanup

---

## **10. Final Git Commit**

Before starting frontend work:
```bash
git add -A
git commit -m "refactor: clean up backend structure before frontend development

- Archive completed backend plans
- Reorganize tests into tests/backend directory
- Remove temporary files (staging, results)
- Update .gitignore
- Create frontend directory structure
- Update documentation"
```

---

## **Next Steps**

1. Run all cleanup tasks above
2. Verify the structure looks clean: `tree -L 2 -I '__pycache__|*.egg-info' .`
3. Ensure backend API is still running and healthy
4. Commit changes to git
5. **Begin React frontend setup** — see [FRONTEND_SETUP.md](FRONTEND_SETUP.md)
