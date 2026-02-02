# Skills Transformation Guide: Converting to Anthropic SKILL.md Format

## Overview

This guide provides step-by-step instructions for transforming a skills folder to comply with Anthropic Claude's official SKILL.md format, enhancing LLM discoverability while maintaining backward compatibility.

## Prerequisites

Before starting, ensure you have:

- Existing skills documentation (numbered or unnumbered files)
- Optional: Cookbook or examples directory with working code
- Optional: README or overview document

## Transformation Procedure

### Phase 1: Analysis and Planning

#### Step 1.1: Inventory Existing Skills

```bash
# Navigate to skills directory
cd /path/to/project/skills

# List all skill files
ls -la

# Count skill files
ls *.md | wc -l
```

**Document:**

- Number of skill files
- Naming convention (e.g., `01-skill-name.md`, `skill-name.md`)
- Presence of README or overview file
- Any existing directory structure

#### Step 1.2: Identify Example Code

```bash
# Check for cookbook or examples
ls -la ../cookbook/
ls -la ../examples/

# List example files
find ../cookbook -name "*.py" -o -name "*.js"
```

**Document:**

- Location of example code
- Mapping between skills and examples
- Shared utilities or dependencies

#### Step 1.3: Review Skill Content

For each skill file, note:

- **Topic/Pattern**: What does it teach?
- **Difficulty**: Beginner, Intermediate, Advanced?
- **Dependencies**: What skills should be learned first?
- **Use Cases**: When should this pattern be used?

#### Step 1.4: Create Implementation Plan

Create [implementation_plan.md](file:///Users/jason/.gemini/antigravity/brain/77f1d1f2-781d-48b6-9937-d34f4dc69549/implementation_plan.md) with:

```markdown
# Skills Transformation Plan

## Current Structure
- [List current files and structure]

## Target Structure
skills/[skill-name]/
├── SKILL.md (main overview)
├── 01-pattern-name.md
├── 02-pattern-name.md
├── ...
└── examples/
    ├── README.md
    ├── 01-example.py
    ├── ...
    └── requirements.txt

## Skill Mapping
| Current File | New Location | Has Example? |
|--------------|--------------|--------------|
| 01-foo.md | [skill-name]/01-foo.md | Yes |
| ... | ... | ... |

## YAML Frontmatter Plan
[For each skill, draft name and description]
```

### Phase 2: Create Directory Structure

#### Step 2.1: Create Main Skill Directory

```bash
# Create directory named after your skill/library
mkdir -p skills/[skill-name]
```

**Naming convention:**

- Use lowercase with hyphens
- Match library/project name
- Examples: `pocketflow-throttled`, `react-patterns`, `api-design`

#### Step 2.2: Copy Skill Files

```bash
# Copy all skill files to new directory
cp 01-*.md [skill-name]/
cp 02-*.md [skill-name]/
# ... etc

# Or copy all at once
cp [0-9][0-9]-*.md [skill-name]/
```

**Verification:**

```bash
# Verify all files copied
ls -la [skill-name]/
diff <(ls *.md | sort) <(ls [skill-name]/*.md | xargs basename -a | sort)
```

### Phase 3: Add YAML Frontmatter

#### Step 3.1: Add Frontmatter to Each Skill File

For each skill file, add YAML frontmatter at the **very top**:

```yaml
---
name: "skill-identifier"
description: "Brief description under 200 chars. Focus on WHEN to use this pattern."
---
```

**Template for adding frontmatter:**

```bash
# For each file, prepend frontmatter
# Example for 01-basic-pattern.md:

cat > temp.md << 'EOF'
---
name: "basic-pattern"
description: "Simple pattern for common use case. Use when you need X and want Y."
---

EOF

cat [skill-name]/01-basic-pattern.md >> temp.md
mv temp.md [skill-name]/01-basic-pattern.md
```

**Best practices for frontmatter:**

1. **Name field:**

   - Use lowercase with hyphens
   - Match filename without number prefix
   - Examples: `basic-node-throttling`, `adaptive-throttling`
2. **Description field:**

   - Keep under 200 characters
   - Start with what it does
   - End with "Use when..." to guide selection
   - Be specific about use cases
   - Examples:
     - ✅ "Fixed rate limits for single-node batch processing. Use when you know exact API rate limits."
     - ❌ "This skill teaches you about throttling"

#### Step 3.2: Verify Frontmatter Format

```bash
# Check first 5 lines of each file
for file in [skill-name]/*.md; do
    echo "=== $file ==="
    head -n 5 "$file"
    echo
done
```

**Common mistakes to avoid:**

- ❌ Extra backticks before `---`
- ❌ Missing closing `---`
- ❌ Quotes not closed properly
- ❌ Description too long (>200 chars)

### Phase 4: Create Main SKILL.md

#### Step 4.1: Draft Main SKILL.md

Create `skills/[skill-name]/SKILL.md`:

```markdown
---
name: "[skill-name]"
description: "High-level description of the entire skill collection. Use when [primary use case]."
---

# [Skill Name Title]

Brief introduction paragraph explaining what this skill collection covers.

## Quick Decision Guide

**Choose the right pattern based on your needs:**

- **[Common problem 1]?** → [01-pattern-name.md](01-pattern-name.md)
- **[Common problem 2]?** → [02-pattern-name.md](02-pattern-name.md)
- **[Common problem 3]?** → [03-pattern-name.md](03-pattern-name.md)
...

## Available Patterns

### Core Skills

1. **[Pattern 1](01-pattern-name.md)** - Brief description
2. **[Pattern 2](02-pattern-name.md)** - Brief description
3. **[Pattern 3](03-pattern-name.md)** - Brief description
...

## When to Use This Skill

Use [skill-name] patterns when you need to:
- [Use case 1]
- [Use case 2]
- [Use case 3]

## Complexity Levels

- **Beginner**: Skills 01, 02
- **Intermediate**: Skills 03, 04, 05
- **Advanced**: Skills 06, 07, 08

## Learning Path

1. Start with **Skill 01** - [Why start here]
2. Read **Skill XX** - [Foundation/architecture]
3. Try **Skill 02** - [Next logical step]
4. Explore **Skill 03** - [Build on previous]
5. Master **Skills 04+** - [Advanced patterns]

## Additional Resources

- **Source Code**: `/path/to/source`
- **Documentation**: `/path/to/docs`
- **Main README**: `/README.md`
```

#### Step 4.2: Customize Decision Guide

The decision guide is **critical** for LLM skill selection. Make it:

- **Question-based**: Start with user's problem
- **Specific**: Direct to exact skill
- **Actionable**: Clear next step

**Example patterns:**

```markdown
## Quick Decision Guide

**Choose based on your situation:**

- **Getting 429 errors?** → [01-basic-throttling.md](01-basic-throttling.md)
- **Don't know the limits?** → [02-adaptive.md](02-adaptive.md)
- **Multiple services?** → [03-shared-limiters.md](03-shared-limiters.md)
```

### Phase 5: Add Examples Directory

#### Step 5.1: Create Examples Directory

```bash
# Create examples directory
mkdir -p [skill-name]/examples
```

#### Step 5.2: Copy Example Files

```bash
# Copy from cookbook or examples
cp ../cookbook/[source]/*.py [skill-name]/examples/

# Rename to match skill numbering
mv [skill-name]/examples/basic_example.py [skill-name]/examples/01-basic-pattern.py
mv [skill-name]/examples/advanced_example.py [skill-name]/examples/02-advanced-pattern.py
# ... etc
```

#### Step 5.3: Copy Supporting Files

```bash
# Copy utilities and dependencies
cp ../cookbook/[source]/utils.py [skill-name]/examples/
cp ../cookbook/[source]/requirements.txt [skill-name]/examples/
```

#### Step 5.4: Create Examples README

Create `skills/[skill-name]/examples/README.md`:

```markdown
# [Skill Name] Examples

Working code examples demonstrating each pattern.

## Quick Start

### Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Running Examples

\`\`\`bash
python 01-basic-pattern.py
python 02-advanced-pattern.py
\`\`\`

## Examples Overview

| Example | Skill | Description |
|---------|-------|-------------|
| [01-basic.py](01-basic.py) | [Skill 01](../01-pattern.md) | Basic usage |
| [02-advanced.py](02-advanced.py) | [Skill 02](../02-pattern.md) | Advanced usage |

## What Each Example Demonstrates

### Example 01: Basic Pattern
- Core concept A
- Core concept B
- Common use case

### Example 02: Advanced Pattern
- Advanced concept A
- Advanced concept B
- Complex use case

## Shared Utilities

- **[utils.py](utils.py)** - Helper functions
- **[requirements.txt](requirements.txt)** - Dependencies

## Customizing for Your Use Case

[Instructions for adapting examples to real-world scenarios]

## Troubleshooting

### Common Issue 1
[Solution]

### Common Issue 2
[Solution]
```

### Phase 6: Update Documentation Cross-References

#### Step 6.1: Update Main SKILL.md with Examples

Add to main [SKILL.md](file:///Users/jason/Documents/GitHub/pocketflow-throttled/skills/pocketflow-throttled/SKILL.md):

```markdown
## Working Examples

See [examples/](examples/) directory for complete, runnable code examples:

- **[01-basic-pattern.py](examples/01-basic-pattern.py)** - Basic usage demo
- **[02-advanced-pattern.py](examples/02-advanced-pattern.py)** - Advanced usage demo
...

See [examples/README.md](examples/README.md) for setup instructions.
```

#### Step 6.2: Update Individual Skill Files

Add "Example Code" section to each skill file **before** "Related Skills":

```markdown
## Example Code

See [examples/01-basic-pattern.py](examples/01-basic-pattern.py) for a complete, runnable example demonstrating [what it demonstrates].
```

**Placement:** Add this section near the end, typically:

1. Main content
2. Best practices
3. Common mistakes
4. **→ Example Code** (new section)
5. Related Skills
6. When NOT to Use

### Phase 7: Verification

#### Step 7.1: Verify Directory Structure

```bash
# Check final structure
tree [skill-name]/

# Expected output:
# [skill-name]/
# ├── SKILL.md
# ├── 01-pattern.md
# ├── 02-pattern.md
# ├── ...
# └── examples/
#     ├── README.md
#     ├── 01-example.py
#     ├── 02-example.py
#     ├── ...
#     ├── utils.py
#     └── requirements.txt
```

#### Step 7.2: Verify YAML Frontmatter

```bash
# Check all files have frontmatter
for file in [skill-name]/*.md; do
    if ! head -n 1 "$file" | grep -q "^---$"; then
        echo "Missing frontmatter: $file"
    fi
done
```

#### Step 7.3: Verify Links

```bash
# Check for broken links (requires markdown-link-check)
npx markdown-link-check [skill-name]/SKILL.md
npx markdown-link-check [skill-name]/*.md
```

#### Step 7.4: Test Examples

```bash
cd [skill-name]/examples

# Install dependencies
pip install -r requirements.txt

# Run each example
python 01-example.py
python 02-example.py
# ... etc
```

### Phase 8: Testing LLM Discovery

#### Step 8.1: Test Skill Discovery

Ask an LLM these questions and verify responses:

**Test 1: General Discovery**

```
Question: "How do I [solve common problem]?"
Expected: LLM discovers [skill-name] via YAML frontmatter
```

**Test 2: Specific Pattern**

```
Question: "Show me an example of [specific pattern]"
Expected: LLM loads specific skill and references example
```

**Test 3: Decision Guide**

```
Question: "I need to [use case], which pattern should I use?"
Expected: LLM uses decision guide to recommend correct skill
```

#### Step 8.2: Verify Progressive Disclosure

The LLM should follow this flow:

1. **Scan frontmatter** → Determine relevance
2. **Load SKILL.md** → Use decision guide
3. **Load specific skill** → Get detailed info
4. **Reference example** → Show implementation

## Common Patterns and Templates

### Pattern 1: Library-Specific Skills

For skills teaching a specific library:

**Directory name:** `[library-name]`
**Main SKILL.md name:** `"[library-name]"`
**Example:** `pocketflow-throttled`, `react-hooks`, `fastapi-patterns`

### Pattern 2: Concept-Based Skills

For skills teaching general concepts:

**Directory name:** `[concept-area]`
**Main SKILL.md name:** `"[concept-area]"`
**Example:** `api-design`, `database-optimization`, `testing-strategies`

### Pattern 3: Technology Stack Skills

For skills covering a full stack:

**Directory name:** `[stack-name]`
**Main SKILL.md name:** `"[stack-name]"`
**Example:** `nextjs-fullstack`, `django-rest`, `express-graphql`

## Checklist

Use this checklist to track progress:

### Planning Phase

- [ ] Inventory existing skills
- [ ] Identify example code
- [ ] Review skill content
- [ ] Create implementation plan

### Structure Phase

- [ ] Create main skill directory
- [ ] Copy skill files
- [ ] Create examples directory
- [ ] Copy example files

### Documentation Phase

- [ ] Add YAML frontmatter to all skills
- [ ] Create main SKILL.md
- [ ] Create examples README
- [ ] Update cross-references

### Verification Phase

- [ ] Verify directory structure
- [ ] Verify YAML frontmatter
- [ ] Verify links work
- [ ] Test examples run
- [ ] Test LLM discovery

## Troubleshooting

### Issue: YAML Frontmatter Errors

**Symptom:** LLM can't discover skills

**Solutions:**

```bash
# Check for extra backticks
grep -n "^\`\`\`" [skill-name]/*.md

# Check for unclosed quotes
grep -n "description:.*\"[^\"]*$" [skill-name]/*.md

# Validate YAML
python -c "import yaml; yaml.safe_load(open('[skill-name]/SKILL.md').read().split('---')[1])"
```

### Issue: Broken Links

**Symptom:** Links don't work in rendered markdown

**Solutions:**

```bash
# Use relative paths, not absolute
# ✅ [example](examples/01-basic.py)
# ❌ [example](/full/path/to/examples/01-basic.py)

# Check all links
find [skill-name] -name "*.md" -exec grep -H "\[.*\](.*)" {} \;
```

### Issue: Examples Don't Run

**Symptom:** Import errors or missing dependencies

**Solutions:**

```bash
# Ensure requirements.txt is complete
pip freeze | grep [library-name] >> requirements.txt

# Test in clean environment
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
python 01-example.py
```

## Best Practices

### 1. YAML Frontmatter

- Keep descriptions under 200 characters
- Focus on "when to use" not "what it is"
- Use consistent naming (lowercase-with-hyphens)

### 2. Main SKILL.md

- Start with clear decision guide
- List skills in logical learning order
- Include complexity levels
- Provide learning path

### 3. Examples

- Use mock APIs when possible (no API keys needed)
- Include comprehensive README
- Match example numbering to skills
- Test all examples before committing

### 4. Documentation

- Add example links to all skills
- Keep cross-references up to date
- Use relative paths for links
- Include "Related Skills" sections

### 5. Backward Compatibility

- Keep original files (copy, don't move)
- Don't break existing references
- Only add, don't remove

## Example Transformation

See the `pocketflow-throttled` transformation as a reference:

**Before:**

```
skills/
├── 00-README.md
├── 01-basic-node-throttling.md
├── 02-adaptive-throttling.md
└── ...
```

**After:**

```
skills/
├── 00-README.md (preserved)
├── 01-basic-node-throttling.md (preserved)
├── ...
└── pocketflow-throttled/
    ├── SKILL.md
    ├── 01-basic-node-throttling.md (with frontmatter)
    ├── 02-adaptive-throttling.md (with frontmatter)
    ├── ...
    └── examples/
        ├── README.md
        ├── 01-basic-node-throttling.py
        ├── ...
        └── requirements.txt
```

## Time Estimates

- **Planning Phase**: 30-60 minutes
- **Structure Phase**: 15-30 minutes
- **Documentation Phase**: 1-2 hours
- **Verification Phase**: 30-60 minutes
- **Total**: 2.5-4.5 hours

Scales with number of skills and complexity of examples.

## Next Steps After Transformation

1. **Test thoroughly** - Run all examples, verify all links
2. **Update main README** - Point to new skill structure
3. **Announce changes** - Document migration in changelog
4. **Monitor usage** - Track LLM skill discovery metrics
5. **Iterate** - Improve based on user feedback

## Resources

- **Anthropic SKILL.md Spec**: [Official documentation]
- **Example Transformations**: See `pocketflow-throttled/`
- **YAML Validator**: https://www.yamllint.com/
- **Markdown Link Checker**: `npx markdown-link-check`

---

**Last Updated**: 2026-02-01
**Version**: 1.0
**Tested With**: pocketflow-throttled skills transformation
