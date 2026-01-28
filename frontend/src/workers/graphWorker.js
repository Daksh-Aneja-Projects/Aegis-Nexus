
/**
 * GraphWorker - Force-Directed Physics Offloading
 * 
 * Logic Level 5 Optimization: Moves O(N^2) repulsion and O(N) spring calculations 
 * to a separate CPU thread, keeping the UI responsive even with 100+ agents.
 */

/* eslint-disable no-restricted-globals */

let nodes = [];
let width = 800;
let height = 600;

// Physics constants
const repulsion = 5000;
const springLength = 100;
const springStrength = 0.03;
const friction = 0.9;
const centerStrength = 0.01;

self.onmessage = (e) => {
    const { type, payload } = e.data;

    switch (type) {
        case 'INIT':
            nodes = payload.nodes.map(n => ({
                ...n,
                x: payload.width / 2 + (Math.random() - 0.5) * 200,
                y: payload.height / 2 + (Math.random() - 0.5) * 200,
                vx: 0,
                vy: 0
            }));
            width = payload.width;
            height = payload.height;
            break;

        case 'UPDATE_NODES':
            // Merge new nodes while preserving positions if they exist
            const newNodes = payload.map(n => {
                const existing = nodes.find(node => node.id === n.id);
                if (existing) {
                    return { ...existing, ...n };
                }
                return {
                    ...n,
                    x: width / 2 + (Math.random() - 0.5) * 50,
                    y: height / 2 + (Math.random() - 0.5) * 50,
                    vx: 0,
                    vy: 0
                };
            });
            nodes = newNodes;
            break;

        case 'TICK':
            step();
            self.postMessage({ type: 'TICK_RESULT', nodes });
            break;

        default:
            break;
    }
};

function step() {
    nodes.forEach(node => {
        let fx = 0;
        let fy = 0;

        // 1. Repulsion (All-pairs)
        nodes.forEach(other => {
            if (node.id === other.id) return;
            const dx = node.x - other.x;
            const dy = node.y - other.y;
            const distSq = dx * dx + dy * dy || 1;
            const force = repulsion / distSq;
            const angle = Math.atan2(dy, dx);
            fx += Math.cos(angle) * force;
            fy += Math.sin(angle) * force;
        });

        // 2. Hub Attraction (Spring)
        if (node.role !== 'hub') {
            const hub = nodes.find(n => n.role === 'hub') || { x: width / 2, y: height / 2 };
            const dx = hub.x - node.x;
            const dy = hub.y - node.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const force = (dist - springLength) * springStrength;
            const angle = Math.atan2(dy, dx);
            fx += Math.cos(angle) * force;
            fy += Math.sin(angle) * force;
        }

        // 3. Center Gravity
        fx += (width / 2 - node.x) * centerStrength;
        fy += (height / 2 - node.y) * centerStrength;

        // Apply forces
        node.vx = (node.vx + fx) * friction;
        node.vy = (node.vy + fy) * friction;
    });

    // Update positions
    nodes.forEach(node => {
        node.x += node.vx;
        node.y += node.vy;

        // Boundary clamping (Soft)
        if (node.x < 0) node.vx += 1;
        if (node.x > width) node.vx -= 1;
        if (node.y < 0) node.vy += 1;
        if (node.y > height) node.vy -= 1;
    });
}
