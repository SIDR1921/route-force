/**
 * A.E.G.I.S. — MoK Loss Interactive Visualisation
 * ═══════════════════════════════════════════════════
 * Renders animated trajectories, step-by-step MoK logic,
 * and MSE vs MoK comparison on HTML5 Canvas.
 */

// ─── Configuration ───
const COLORS = {
    gt: '#00D2FF',
    mode1: '#6C5CE7',
    mode2: '#FDCB6E',
    mode3: '#E17055',
    winner: '#00B894',
    grid: 'rgba(255,255,255,0.03)',
    gridLine: 'rgba(255,255,255,0.06)',
    agent: '#e8e8f0',
};

const MODE_COLORS = [COLORS.mode1, COLORS.mode2, COLORS.mode3];

// ─── Bézier Math ───
function bezier(t, P0, P1, P2, P3) {
    const u = 1 - t;
    return {
        x: u*u*u*P0.x + 3*u*u*t*P1.x + 3*u*t*t*P2.x + t*t*t*P3.x,
        y: u*u*u*P0.y + 3*u*u*t*P1.y + 3*u*t*t*P2.y + t*t*t*P3.y,
    };
}

function generateBezierPath(pts, steps = 30) {
    const path = [];
    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        path.push(bezier(t, pts[0], pts[1], pts[2], pts[3]));
    }
    return path;
}

// Generate sample scenario: pedestrian at intersection
function generateScenario() {
    const cx = 400, cy = 320;

    // History: walking from bottom-left
    const history = [];
    for (let i = 0; i < 8; i++) {
        history.push({x: cx - 120 + i * 15, y: cy + 60 - i * 8});
    }

    // Ground truth: continues slightly left (actually goes left)
    const gtCtrl = [
        {x: cx, y: cy},
        {x: cx - 30, y: cy - 60},
        {x: cx - 80, y: cy - 120},
        {x: cx - 130, y: cy - 180},
    ];
    const gt = generateBezierPath(gtCtrl);

    // Mode 1: Goes left (closest to GT — will win)
    const m1Ctrl = [
        {x: cx, y: cy},
        {x: cx - 20, y: cy - 50},
        {x: cx - 70, y: cy - 110},
        {x: cx - 120, y: cy - 170},
    ];

    // Mode 2: Goes straight ahead
    const m2Ctrl = [
        {x: cx, y: cy},
        {x: cx + 10, y: cy - 70},
        {x: cx + 20, y: cy - 140},
        {x: cx + 30, y: cy - 200},
    ];

    // Mode 3: Turns right
    const m3Ctrl = [
        {x: cx, y: cy},
        {x: cx + 50, y: cy - 40},
        {x: cx + 120, y: cy - 60},
        {x: cx + 190, y: cy - 80},
    ];

    const modes = [
        generateBezierPath(m1Ctrl),
        generateBezierPath(m2Ctrl),
        generateBezierPath(m3Ctrl),
    ];

    // Compute losses (average distance to GT)
    const losses = modes.map(mode => {
        let sum = 0;
        for (let i = 0; i < gt.length; i++) {
            const dx = mode[i].x - gt[i].x;
            const dy = mode[i].y - gt[i].y;
            sum += Math.sqrt(dx*dx + dy*dy);
        }
        return sum / gt.length;
    });

    return { history, gt, modes, losses, cx, cy };
}

// ─── Particle Background ───
function initParticles() {
    const canvas = document.getElementById('particles-bg');
    const ctx = canvas.getContext('2d');
    let W, H;
    const particles = [];
    const NUM = 60;

    function resize() {
        W = canvas.width = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }

    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < NUM; i++) {
        particles.push({
            x: Math.random() * W,
            y: Math.random() * H,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            r: Math.random() * 1.5 + 0.5,
            alpha: Math.random() * 0.4 + 0.1,
        });
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = W;
            if (p.x > W) p.x = 0;
            if (p.y < 0) p.y = H;
            if (p.y > H) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(108, 92, 231, ${p.alpha})`;
            ctx.fill();
        });

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(108, 92, 231, ${0.08 * (1 - dist/120)})`;
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(draw);
    }

    draw();
}

// ─── Main Trajectory Canvas ───
class TrajectoryVisualizer {
    constructor() {
        this.canvas = document.getElementById('trajectory-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlay = document.getElementById('canvas-overlay');
        this.scenario = generateScenario();
        this.animationStep = 0;
        this.maxSteps = 31;
        this.isPlaying = false;
        this.animFrame = null;
        this.currentStep = 0; // 0=idle, 1=compare, 2=find winner, 3=reward, 4=diversify

        this.setupButtons();
        this.drawStatic();
    }

    setupButtons() {
        document.getElementById('btn-play').addEventListener('click', () => this.play());
        document.getElementById('btn-reset').addEventListener('click', () => this.reset());
        document.getElementById('btn-step').addEventListener('click', () => this.stepForward());
    }

    drawGrid() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Background
        ctx.fillStyle = 'rgba(5, 5, 20, 1)';
        ctx.fillRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = COLORS.gridLine;
        ctx.lineWidth = 0.5;
        for (let x = 0; x < w; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
    }

    drawHistory() {
        const ctx = this.ctx;
        const hist = this.scenario.history;

        // Trail
        ctx.beginPath();
        ctx.moveTo(hist[0].x, hist[0].y);
        for (let i = 1; i < hist.length; i++) {
            ctx.lineTo(hist[i].x, hist[i].y);
        }
        ctx.strokeStyle = 'rgba(232, 232, 240, 0.4)';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Dots
        hist.forEach((p, i) => {
            const alpha = 0.3 + (i / hist.length) * 0.7;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(232, 232, 240, ${alpha})`;
            ctx.fill();
        });

        // Agent marker
        const last = hist[hist.length - 1];
        ctx.beginPath();
        ctx.arc(last.x, last.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.agent;
        ctx.fill();
        ctx.strokeStyle = 'rgba(108, 92, 231, 0.6)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = 'rgba(232, 232, 240, 0.6)';
        ctx.font = '11px Inter';
        ctx.fillText('Agent (t=0)', last.x + 14, last.y + 4);
    }

    drawPath(path, color, steps, lineWidth = 2.5, dashed = false) {
        const ctx = this.ctx;
        const n = Math.min(steps, path.length);
        if (n < 2) return;

        ctx.beginPath();
        ctx.moveTo(path[0].x, path[0].y);
        for (let i = 1; i < n; i++) {
            ctx.lineTo(path[i].x, path[i].y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        if (dashed) ctx.setLineDash([6, 3]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Endpoint dot
        if (n > 1) {
            const end = path[n - 1];
            ctx.beginPath();
            ctx.arc(end.x, end.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        }
    }

    drawGlow(path, color, steps) {
        const ctx = this.ctx;
        const n = Math.min(steps, path.length);
        if (n < 2) return;

        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.moveTo(path[0].x, path[0].y);
        for (let i = 1; i < n; i++) {
            ctx.lineTo(path[i].x, path[i].y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.restore();
    }

    drawStatic() {
        this.drawGrid();
        this.drawHistory();
    }

    drawFrame(step) {
        this.drawGrid();
        this.drawHistory();

        const { gt, modes, losses } = this.scenario;
        const bestMode = losses.indexOf(Math.min(...losses));

        // Draw GT
        this.drawPath(gt, COLORS.gt, step, 3);

        // Draw modes
        modes.forEach((mode, i) => {
            const isWinner = (i === bestMode && step >= this.maxSteps);
            const color = isWinner ? COLORS.winner : MODE_COLORS[i];
            const width = isWinner ? 3.5 : 2;

            if (isWinner && step >= this.maxSteps) {
                this.drawGlow(mode, COLORS.winner, step);
            }
            this.drawPath(mode, color, step, width);
        });

        // Update loss bars
        if (step > 0) {
            const maxLoss = Math.max(...losses);
            losses.forEach((loss, i) => {
                const pct = (loss / maxLoss) * 100;
                document.getElementById(`bar-${i+1}`).style.width = `${pct}%`;
                document.getElementById(`loss-${i+1}`).textContent = loss.toFixed(2);

                if (step >= this.maxSteps && i === bestMode) {
                    document.getElementById(`bar-${i+1}`).classList.add('winner');
                }
            });
        }

        // Update winner label
        if (step >= this.maxSteps) {
            document.getElementById('winner-label').querySelector('span:last-child').textContent = `Winner: Mode ${bestMode + 1}`;
            document.getElementById('gradient-status').querySelector('span').textContent = 
                `Only Mode ${bestMode + 1} receives gradient! Others explore freely.`;
        }
    }

    play() {
        this.overlay.classList.add('hidden');

        if (this.isPlaying) return;
        this.isPlaying = true;
        this.animationStep = 0;

        // Highlight step cards
        this.highlightStep(1);

        const animate = () => {
            if (this.animationStep <= this.maxSteps) {
                this.drawFrame(this.animationStep);
                this.animationStep++;

                // Update step highlights
                if (this.animationStep === Math.floor(this.maxSteps * 0.3)) this.highlightStep(2);
                if (this.animationStep === Math.floor(this.maxSteps * 0.6)) this.highlightStep(3);
                if (this.animationStep >= this.maxSteps) this.highlightStep(4);

                this.animFrame = setTimeout(() => requestAnimationFrame(animate), 60);
            } else {
                this.isPlaying = false;
            }
        };

        requestAnimationFrame(animate);
    }

    stepForward() {
        this.overlay.classList.add('hidden');
        if (this.animationStep <= this.maxSteps) {
            this.animationStep++;
            this.drawFrame(this.animationStep);
        }
    }

    reset() {
        this.isPlaying = false;
        if (this.animFrame) clearTimeout(this.animFrame);
        this.animationStep = 0;
        this.overlay.classList.remove('hidden');

        // Reset UI
        [1,2,3].forEach(i => {
            document.getElementById(`bar-${i}`).style.width = '0%';
            document.getElementById(`bar-${i}`).classList.remove('winner');
            document.getElementById(`loss-${i}`).textContent = '—';
        });
        document.getElementById('winner-label').querySelector('span:last-child').textContent = 'Winner: —';
        document.getElementById('gradient-status').querySelector('span').textContent = 'Waiting...';
        document.querySelectorAll('.step-card').forEach(c => c.classList.remove('active'));

        this.scenario = generateScenario();
        this.drawStatic();
    }

    highlightStep(step) {
        document.querySelectorAll('.step-card').forEach(c => c.classList.remove('active'));
        const card = document.querySelector(`.step-card[data-step="${step}"]`);
        if (card) card.classList.add('active');
    }
}

// ─── MSE vs MoK Comparison Canvases ───
function drawComparisonMSE() {
    const canvas = document.getElementById('mse-canvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;

    // Background
    ctx.fillStyle = 'rgba(5, 5, 20, 1)';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    for (let x = 0; x < w; x += 25) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke(); }
    for (let y = 0; y < h; y += 25) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }

    const cx = 175, cy = 200;

    // Agent
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI*2);
    ctx.fillStyle = COLORS.agent;
    ctx.fill();

    // GT path (goes left)
    const gtPath = generateBezierPath([
        {x:cx, y:cy}, {x:cx-30, y:cy-40}, {x:cx-60, y:cy-80}, {x:cx-90, y:cy-140}
    ]);

    // All 3 modes collapse to middle
    const mean = generateBezierPath([
        {x:cx, y:cy}, {x:cx, y:cy-45}, {x:cx, y:cy-90}, {x:cx, y:cy-140}
    ]);

    // Draw GT
    ctx.beginPath();
    gtPath.forEach((p,i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.strokeStyle = COLORS.gt;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Draw 3 collapsed modes (all nearly identical)
    [0, 1, 2].forEach(i => {
        const offset = (i - 1) * 3;
        ctx.beginPath();
        mean.forEach((p,j) => j === 0 ? ctx.moveTo(p.x + offset, p.y) : ctx.lineTo(p.x + offset, p.y));
        ctx.strokeStyle = MODE_COLORS[i];
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;
        ctx.stroke();
        ctx.globalAlpha = 1;
    });

    // "COLLAPSED!" label
    ctx.fillStyle = 'rgba(225, 112, 85, 0.8)';
    ctx.font = 'bold 11px Inter';
    ctx.fillText('All modes collapsed!', cx - 10, 40);
}

function drawComparisonMoK() {
    const canvas = document.getElementById('mok-canvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;

    ctx.fillStyle = 'rgba(5, 5, 20, 1)';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    for (let x = 0; x < w; x += 25) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke(); }
    for (let y = 0; y < h; y += 25) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }

    const cx = 175, cy = 200;

    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI*2);
    ctx.fillStyle = COLORS.agent;
    ctx.fill();

    // GT
    const gt = generateBezierPath([
        {x:cx, y:cy}, {x:cx-30, y:cy-40}, {x:cx-60, y:cy-80}, {x:cx-90, y:cy-140}
    ]);

    // Three diverse modes
    const paths = [
        generateBezierPath([{x:cx,y:cy}, {x:cx-25,y:cy-45}, {x:cx-55,y:cy-85}, {x:cx-85,y:cy-135}]),
        generateBezierPath([{x:cx,y:cy}, {x:cx+5,y:cy-50}, {x:cx+15,y:cy-95}, {x:cx+20,y:cy-145}]),
        generateBezierPath([{x:cx,y:cy}, {x:cx+40,y:cy-25}, {x:cx+90,y:cy-40}, {x:cx+140,y:cy-55}]),
    ];

    // GT
    ctx.beginPath();
    gt.forEach((p,i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.strokeStyle = COLORS.gt;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Modes with glow on winner
    paths.forEach((path, i) => {
        if (i === 0) {
            ctx.save();
            ctx.shadowColor = COLORS.winner;
            ctx.shadowBlur = 10;
        }

        ctx.beginPath();
        path.forEach((p,j) => j === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
        ctx.strokeStyle = i === 0 ? COLORS.winner : MODE_COLORS[i];
        ctx.lineWidth = i === 0 ? 3 : 2;
        ctx.stroke();

        if (i === 0) ctx.restore();
    });

    // Labels
    ctx.fillStyle = COLORS.winner;
    ctx.font = 'bold 10px Inter';
    ctx.fillText('🏆 Winner', paths[0][paths[0].length-1].x - 40, paths[0][paths[0].length-1].y - 8);
    
    ctx.fillStyle = 'rgba(253, 203, 110, 0.7)';
    ctx.fillText('Exploring ↑', paths[1][paths[1].length-1].x + 5, paths[1][paths[1].length-1].y - 5);
    
    ctx.fillStyle = 'rgba(225, 112, 85, 0.7)';
    ctx.fillText('Exploring →', paths[2][paths[2].length-1].x + 5, paths[2][paths[2].length-1].y + 4);
}

// ─── Initialize ───
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    new TrajectoryVisualizer();
    drawComparisonMSE();
    drawComparisonMoK();
});
