import { afterEach, describe, expect, it, vi } from 'vitest';

import { getTemplate, getTemplatesByCategory, listTemplates, templates } from './templates.js';

describe('templates', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('exports predefined templates', () => {
    expect(Array.isArray(templates)).toBe(true);
    expect(templates.length).toBeGreaterThan(0);
  });

  it('each template has required fields', () => {
    for (const tmpl of templates) {
      expect(tmpl.id).toBeDefined();
      expect(tmpl.name).toBeDefined();
      expect(tmpl.description).toBeDefined();
      expect(tmpl.category).toBeDefined();
      expect(['integration', 'experiment', 'analysis', 'custom']).toContain(tmpl.category);
      expect(typeof tmpl.createNodes).toBe('function');
    }
  });

  describe('getTemplate', () => {
    it('returns template by id', () => {
      const tmpl = getTemplate('full-integration');
      expect(tmpl).toBeDefined();
      expect(tmpl!.name).toBe('Full Integration');
    });

    it('returns undefined for unknown id', () => {
      expect(getTemplate('nonexistent')).toBeUndefined();
    });
  });

  describe('getTemplatesByCategory', () => {
    it('returns templates matching category', () => {
      const integrationTmpls = getTemplatesByCategory('integration');
      expect(integrationTmpls.length).toBeGreaterThan(0);
      for (const tmpl of integrationTmpls) {
        expect(tmpl.category).toBe('integration');
      }
    });

    it('returns empty array for unknown category', () => {
      const result = getTemplatesByCategory('unknown-category');
      expect(result).toEqual([]);
    });

    it('returns analysis templates', () => {
      const analysisTmpls = getTemplatesByCategory('analysis');
      expect(analysisTmpls.length).toBeGreaterThan(0);
    });

    it('returns experiment templates', () => {
      const experimentTmpls = getTemplatesByCategory('experiment');
      expect(experimentTmpls.length).toBeGreaterThan(0);
    });
  });

  describe('listTemplates', () => {
    it('returns summary objects for all templates', () => {
      const list = listTemplates();
      expect(list).toHaveLength(templates.length);
      for (const item of list) {
        expect(item).toHaveProperty('id');
        expect(item).toHaveProperty('name');
        expect(item).toHaveProperty('description');
        expect(item).toHaveProperty('category');
        expect(Object.keys(item)).toEqual(['id', 'name', 'description', 'category']);
      }
    });
  });

  describe('createNodes', () => {
    it('full-integration template creates workflow nodes', () => {
      const mockBridge = {} as any;
      const tmpl = getTemplate('full-integration')!;
      const nodes = tmpl.createNodes(mockBridge);
      expect(Array.isArray(nodes)).toBe(true);
      expect(nodes.length).toBeGreaterThan(0);
      // Should have analyze, research, mapping, patch, validation, report
      expect(nodes.map(n => n.id)).toEqual([
        'analyze', 'research', 'mapping', 'patch', 'validation', 'report',
      ]);
    });

    it('quick-analyze template creates 2 nodes', () => {
      const mockBridge = {} as any;
      const tmpl = getTemplate('quick-analyze')!;
      const nodes = tmpl.createNodes(mockBridge);
      expect(nodes).toHaveLength(2);
      expect(nodes[0].id).toBe('analyze');
      expect(nodes[1].id).toBe('suggest');
    });

    it('safe-integration template includes critic and validation gate', () => {
      const mockBridge = {} as any;
      const tmpl = getTemplate('safe-integration')!;
      const nodes = tmpl.createNodes(mockBridge);
      const nodeIds = nodes.map(n => n.id);
      expect(nodeIds).toContain('critic');
      expect(nodeIds).toContain('validation-gate');
    });
  });
});
