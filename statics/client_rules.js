// Client dev rules UI script
// Mirrors the dev rules editor used in rules.html. Serializes rules into the hidden
// input `client_dev_rules_json` on form submit.

(function(){
    console.debug('client_rules.js: init', typeof window !== 'undefined');
    const init = window.__DPY_INIT_CLIENT_RULES || {};
    const incomingParams = init.available_rule_keys || [];
    const existingRules = init.client_dev_rules || [];
    const rulesContainer = document.getElementById('dev-rules-container') || document.getElementById('client-rules-container');
    const addDevRuleBtn = document.getElementById('add-dev-rule-btn') || document.getElementById('add-client-rule');
    const form = document.getElementById('client-form');

    function makeConditionOptions(rule) {
        let html = '<option value="">Always Apply (Default)</option>';
        for (const p of incomingParams) {
            html += `<option value="${p}" ${rule && rule.condition_param==p ? 'selected' : ''}>IF ${p}</option>`;
        }
        return html;
    }

    function makeTargetOptions(rule) {
        let html = '';
        for (const p of incomingParams) {
            html += `<option value="${p}" ${rule && rule.target_param==p ? 'selected' : ''}>${p}</option>`;
        }
        return html;
    }

    function createDevRuleRow(rule) {
        const rc = document.getElementById('dev-rules-container') || document.getElementById('client-rules-container');
        if (!rc) {
            console.error('Dev rules container not found; cannot add rule row');
            return;
        }
        const row = document.createElement('div');
        row.className = 'flex items-center gap-2 p-2 bg-gray-50 rounded';
        row.innerHTML = `
            <select name="dev_condition_param" class="p-2 border rounded w-1/4 text-sm">${makeConditionOptions(rule)}</select>
            <span class="text-gray-500 equals-span"></span>
            <input name="dev_condition_value" value="${rule && rule.condition_value ? rule.condition_value : ''}" placeholder="contains this value..." class="p-2 border rounded w-1/4 text-sm">
            <span class="text-gray-500">THEN</span>
            <select name="dev_target_param" class="p-2 border rounded w-1/4 text-sm">${makeTargetOptions(rule)}</select>
            <input name="dev_override_value" value="${rule && rule.override_value ? rule.override_value : ''}" placeholder="to this new value" class="p-2 border rounded flex-grow text-sm">
            <button type="button" class="remove-dev-rule px-3 py-2 bg-red-600 text-white rounded text-sm w-20 flex-shrink-0">Remove</button>
        `;
        try {
            rc.appendChild(row);
        } catch (err) {
            console.error('Failed to append dev rule row', err);
            return;
        }
        // wire condition change display
        const condSel = row.querySelector('select[name="dev_condition_param"]');
        condSel.addEventListener('change', function(){
            const val = this.value;
            const valInput = row.querySelector('input[name="dev_condition_value"]');
            const eq = row.querySelector('.equals-span');
            if (val) {
                if (valInput) valInput.style.display = 'block';
                if (eq) eq.style.display = 'inline';
            } else {
                if (valInput) valInput.style.display = 'none';
                if (eq) eq.style.display = 'none';
            }
        });
        // initial show/hide
        try { condSel.dispatchEvent(new Event('change')); } catch(e){}
        // update hidden input after adding row
        try { serializeRulesToHidden(); } catch(e){}
    }

    function serializeRulesToHidden() {
        try {
            const rc = document.getElementById('dev-rules-container') || document.getElementById('client-rules-container');
            const rows = Array.from((rc && rc.querySelectorAll('div')) || []);
            const out = [];
            for (const row of rows) {
                const conditionParam = row.querySelector('select[name="dev_condition_param"]');
                const conditionValue = row.querySelector('input[name="dev_condition_value"]');
                const target = row.querySelector('select[name="dev_target_param"]');
                const overrideVal = row.querySelector('input[name="dev_override_value"]');
                out.push({ endpoint_id: null, condition_param: conditionParam ? conditionParam.value : null, condition_value: conditionValue ? conditionValue.value : null, target_param: target ? target.value : null, override_value: overrideVal ? overrideVal.value : null, active: true });
            }
            const hidden = document.getElementById('client_dev_rules_json');
            if (hidden) hidden.value = JSON.stringify(out);
            return out;
        } catch(e) { console.error('client_rules.js: serializeRulesToHidden error', e); return []; }
    }

    // populate existing rules or a blank one (only if the client actually has Dev Mode / dev-rules enabled)
    try {
        const rcCheck = document.getElementById('dev-rules-container') || document.getElementById('client-rules-container');
        if (rcCheck) {
            if (Array.isArray(existingRules) && existingRules.length) {
                // populate real existing rules
                for (const r of existingRules) createDevRuleRow(r);
            } else {
                // Only create a blank rule row when Dev Mode and Dev Rules are actually enabled.
                const devRulesCheckbox = document.querySelector('input[name="dev_rules_enabled"]');
                const devModeHidden = document.querySelector('input[name="is_dev_client"]');
                let devModeOn = false;
                try {
                    // Removed reliance on is_dev_client
                    // devModeOn = devModeHidden && ['1','true','on','True'].includes(String(devModeHidden.value));
                } catch(e) { devModeOn = false; }

                if (devRulesCheckbox && devRulesCheckbox.checked) {
                    createDevRuleRow({});
                }
            }
        } else {
            // container not present yet; try to populate later when user expands Dev Mode
            // no-op here
        }
    } catch(e) { console.error('Error populating existing dev rules', e); }

    // add button: use event delegation so it works even if Alpine re-renders or timing differs
    document.addEventListener('click', function(e){
        const target = e.target || e.srcElement;
        if (!target) return;
        // match the add buttons by id or by class
        if (target.closest && (target.closest('#add-dev-rule-btn') || target.closest('#add-client-rule'))) {
            console.debug('client_rules.js: add button clicked');
            // ensure rulesContainer exists (Alpine might change DOM timing)
            const rc = document.getElementById('dev-rules-container') || document.getElementById('client-rules-container');
            if (!rc) return;
            createDevRuleRow({});
            try { serializeRulesToHidden(); } catch(e){}
        }
    });

        // When the user enables the "Activate development rules" checkbox,
        // ensure at least one blank rule row is present so the UI is ready to edit.
        // Avoid creating a blank rule on arbitrary button clicks (which caused
        // accidental empty rules to be serialized on save).
        (function(){
            const rcGetter = () => document.getElementById('dev-rules-container') || document.getElementById('client-rules-container');
            const devRulesCheckbox = document.querySelector('input[name="dev_rules_enabled"]');
            if (devRulesCheckbox) {
                devRulesCheckbox.addEventListener('change', function(){
                    try {
                        const rc = rcGetter();
                        if (!rc) return;
                        if (this.checked && rc.children.length === 0) {
                            // Create a blank editable row and serialize immediately.
                            createDevRuleRow({});
                            try { serializeRulesToHidden(); } catch(e){}
                        }
                    } catch(e) {
                        // best-effort
                    }
                });
            }
        })();

    // event delegation for remove
    // event delegation for remove (lookup container at runtime)
    document.addEventListener('click', function(e){
        if (e.target && e.target.classList && e.target.classList.contains('remove-dev-rule')) {
            console.debug('client_rules.js: remove button clicked');
            const r = e.target.closest('div');
            if (r) r.remove();
            try { serializeRulesToHidden(); } catch(e){}
        }
    });

    // update serialization whenever user types or changes selects inside rules container
    document.addEventListener('input', function(e){
        const rc = document.getElementById('dev-rules-container') || document.getElementById('client-rules-container');
        if (!rc) return;
        if (rc.contains(e.target)) {
            try { serializeRulesToHidden(); } catch(e){}
        }
    });
    document.addEventListener('change', function(e){
        const rc = document.getElementById('dev-rules-container') || document.getElementById('client-rules-container');
        if (!rc) return;
        if (rc.contains(e.target)) {
            try { serializeRulesToHidden(); } catch(e){}
        }
    });

    // Validate dev rules and serialize to client_dev_rules_json on client form submit
    if (form) {
        form.addEventListener('submit', function(e){
            try {
                const devRulesCheckbox = document.querySelector('input[name="dev_rules_enabled"]');
                const devModeHidden = document.querySelector('input[name="is_dev_client"]') || document.querySelector('input[name="is_dev_client"]');
                const devModeOn = devModeHidden && ['1','true','on'].includes(String(devModeHidden.value));
                const rulesEnabled = devRulesCheckbox && devRulesCheckbox.checked;

                console.debug('client_rules.js: submit', { devModeHiddenValue: devModeHidden && devModeHidden.value, devModeOn, rulesEnabled });

                if (rulesEnabled) {
                    const rc = document.getElementById('dev-rules-container') || document.getElementById('client-rules-container');
                    const rows = Array.from((rc && rc.querySelectorAll('div')) || []);
                    let anyValid = false;
                    const invalids = [];
                    const out = [];
                    for (const row of rows) {
                        const conditionParam = row.querySelector('select[name="dev_condition_param"]');
                        const conditionValue = row.querySelector('input[name="dev_condition_value"]');
                        const target = row.querySelector('select[name="dev_target_param"]');
                        const overrideVal = row.querySelector('input[name="dev_override_value"]');

                        const hasTarget = target && target.value && String(target.value).trim();
                        const hasOverride = overrideVal && overrideVal.value && String(overrideVal.value).trim();

                        let conditionSatisfied = true;
                        if (conditionParam && conditionParam.value) {
                            if (!(conditionValue && conditionValue.value && String(conditionValue.value).trim())) {
                                conditionSatisfied = false;
                                invalids.push(conditionValue || conditionParam);
                            }
                        }

                        if (hasTarget && hasOverride && conditionSatisfied) anyValid = true;

                        if (!hasTarget) invalids.push(target);
                        if (!hasOverride) invalids.push(overrideVal);

                        out.push({ endpoint_id: null, condition_param: conditionParam ? conditionParam.value : null, condition_value: conditionValue ? conditionValue.value : null, target_param: target ? target.value : null, override_value: overrideVal ? overrideVal.value : null, active: true });
                    }
                    if (!anyValid || invalids.length > 0) {
                        // prevent submit and show simple alert (mirrors rules.html behavior)
                        e.preventDefault();
                        for (const inv of invalids) if (inv) inv.classList.add('border-red-600');
                        console.debug('client_rules.js: submit prevented - no valid rules', { rowsCount: rows.length, invalidsCount: invalids.length, out });
                        return false;
                    }

                    // serialize
                    const hidden = document.getElementById('client_dev_rules_json');
                    if (hidden) {
                        hidden.value = JSON.stringify(out);
                        console.debug('client_rules.js: serialized rules into hidden input', hidden.value);
                    }
                } else {
                    // No dev rules enabled; ensure we send an empty array so server clears rules if appropriate
                    const hidden = document.getElementById('client_dev_rules_json');
                    if (hidden) {
                        hidden.value = JSON.stringify([]);
                        console.debug('client_rules.js: dev rules disabled at submit; wrote empty array');
                    }
                }
            } catch(err) {
                console.error('client_rules.js: submit handler error', err);
                // best effort: ensure hidden input exists and set empty array to avoid losing rules accidentally
                try { const hidden = document.getElementById('client_dev_rules_json'); if (hidden && !hidden.value) hidden.value = JSON.stringify([]); } catch(e){}
            }
        });
    }

    // Expose a helper so external scripts (client submit handler) can force
    // serialization of current in-memory rules into the hidden input before
    // performing validation or submitting via fetch.
    try {
        window.__DPY_SERIALIZE_RULES = serializeRulesToHidden;
    } catch (e) {
        /* best-effort */
    }

})();
